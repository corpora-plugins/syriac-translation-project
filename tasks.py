import re
import json
import unicodedata
import traceback
from django.template import Template, Context
from google import genai
from google.genai import types
from corpus import Job

LINE_CHUNK_SIZE = 30

# Approximate Gemini API pricing in USD per 1,000,000 tokens. "Thinking" (thoughts)
# tokens are billed at the output rate. Update these as Google changes pricing:
# https://ai.google.dev/gemini-api/docs/pricing
MODEL_PRICING = {
    'gemini-3.5-flash': {'input': 1.50, 'output': 9.00},
    'gemini-3.6-flash': {'input': 0.75, 'output': 3.75},
}
DEFAULT_PRICING = {'input': 0.30, 'output': 2.50}

SYRIAC_RE = re.compile(r"[\u0700-\u074F\u0860-\u086F]")
SYRIAC_MARKS_RE = re.compile(r"[\u0730-\u074A]")
INVISIBLE_CHARS = str.maketrans("", "", "\ufeff\u200e\u200f\u061c") # deleting all characters in third parameter

LEXICAL_NOTES_RE = re.compile(r'<LEXICAL_NOTES\s+number=["\']?(\d+)["\']?\s*>(.*?)</LEXICAL_NOTES>', re.DOTALL)
TRANSLATOR_NOTES_RE = re.compile(r'<TRANSLATION_NOTES\s+number=["\']?(\d+)["\']?\s*>(.*?)</TRANSLATION_NOTES>', re.DOTALL)
TRANSLATIONS_RE = re.compile(r'<TRANSLATION\s+number=["\']?(\d+)["\']?\s*>(.*?)</TRANSLATION>', re.DOTALL)
REVISIONS_RE = re.compile(r'<REVISION\s+number=["\']?(\d+)["\']?\s*>(.*?)</REVISION>', re.DOTALL)
reviewer_notes_RE = re.compile(r'<REVISION_NOTE\s+number=["\']?(\d+)["\']?\s*>(.*?)</REVISION_NOTE>', re.DOTALL)


REGISTRY = {
    "Translate Syriac Text": {
        "version": "0.6",
        "jobsite_type": "HUEY",
        "track_provenance": True,
        "create_report": True,
        "content_type": "SyriacText",
        "configuration": {
            "parameters": {
                "model": {
                    "label": "Gemini Model",
                    "value": "gemini-3.7-flash",
                    "type": "choice",
                    "choices": [
                        "gemini-3.7-flash",
                        'gemini-3.6-flash',
                        'gemini-3.5-flash',
                    ]
                },
                "thinking_level": {
                    "label": "Thinking Level",
                    "value": "high",
                    "type": "choice",
                    "choices": [
                        'high',
                        'medium',
                        'low',
                        'minimal',
                    ]
                },
                "scope": {
                    "label": "Scope",
                    "value": "full",
                    "type": "choice",
                    "choices": [
                        'full',
                        'rare word detection only',
                        'lexical notes only',
                        'translations only',
                        'revisions only'
                    ]
                }
            },
        },
        "module": 'plugins.syriac_translation_project.tasks',
        "functions": ['translate_syriac_text']
     }
}


def translate_syriac_text(job_id):
    job = Job(job_id)
    corpus = job.corpus
    model = job.get_param_value('model')
    thinking_level = job.get_param_value('thinking_level')
    scope = job.get_param_value('scope')
    total_lines = 0
    usage_totals = new_usage_totals()

    job.set_status('running')

    # check if lines for this text already exist
    if scope in ['full', 'rare word detection only']:
        # break text into lines
        total_lines = create_syriac_lines(job)
    else:
        total_lines = corpus.get_content('SyriacLine', {'syriac_text': job.content.id}).count()

    if total_lines:
        # instantiate our Gemini client
        gemini_client = genai.Client(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=5.0,
                    exp_base=2.0,
                    http_status_codes=[429, 500, 502, 503, 504],
                )
            )
        )

        # set up the thinking level config
        thinking_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level,
            )
        )

        if scope in ['full', 'lexical notes only']:
            # use AI to create lexical notes
            create_lexical_notes(job, gemini_client, model, thinking_config, usage_totals)

        if scope in ['full', 'translations only']:
            # use AI to translate the lines
            create_translations(job, gemini_client, model, thinking_config, usage_totals)

        if scope in ['full', 'revisions only']:
            # use AI to review translations
            review_lines(job, gemini_client, model, thinking_config, usage_totals)

        # report an estimate of what the Gemini API calls for this job cost
        if usage_totals['api_calls']:
            job.report(format_cost_report(model, usage_totals, "Estimated total Gemini API cost for this job"))

    else:
        job.report(f'Either rare word detection failed or no pre-existing lines were found for performing {scope}!')

    job.content.job_report = f"/corpus/{job.corpus.id}/get-file/?path=job_reports/{job_id}.txt"
    job.content.save()
    job.complete(status='complete')


def create_syriac_lines(job):
    corpus = job.corpus
    syriac_text = job.content
    stage = 1

    # clear any existing lines
    existing_syriac_lines = corpus.get_content('SyriacLine', {'syriac_text': syriac_text.id})
    for existing_syriac_line in existing_syriac_lines:
        existing_syriac_line.delete()

    # split original text into lines
    original_lines = syriac_text.original.splitlines()
    original_lines = [line.strip() for line in original_lines if line]
    total_lines = len(original_lines)
    line_number = 1

    # build new lines
    for original_line in original_lines:

        syriac_line = corpus.get_content('SyriacLine')
        syriac_line.syriac_text = syriac_text.id
        syriac_line.original = original_line
        syriac_line.line_number = line_number

        # hunt for rare words and associate them
        words = original_line.split()
        for word in words:
            normalized = normalize_syriac_word(word)
            rare_word = corpus.get_content('RareWord', {'normalized': normalized}, single_result=True)
            if rare_word:
                syriac_line.rare_words.append(word)
                syriac_line.rare_words_normalized.append(rare_word.id)

        syriac_line.save()
        line_number += 1

        if (line_number - 1) % LINE_CHUNK_SIZE == 0:
            report_percent_complete(job, stage, line_number - 1, total_lines)

    job.report(f"Created {total_lines} lines of Syriac for processing.")

    # return the total number of lines
    return total_lines


def create_lexical_notes(job, gemini_client, model, thinking_config, usage_totals=None):
    corpus = job.corpus
    syriac_text = job.content
    stage = 2
    lexical_notes_created = 0

    # grab our lexical notes prompt
    prompt = corpus.get_content('Prompt', {'name': 'Lexical Investigator'}, single_result=True)

    # grab syriac lines
    syriac_lines = corpus.get_content('SyriacLine', {'syriac_text': syriac_text.id}).order_by('line_number')

    # run them through model and get deliverables
    line_deliverables = run_lines_through_model(
        job,
        stage,
        syriac_lines,
        gemini_client,
        model,
        thinking_config,
        prompt,
        {},
        usage_totals=usage_totals
    )

    for line in syriac_lines:
        if line.line_number in line_deliverables:
            line.lexical_notes = [note for note in line_deliverables[line.line_number]['lexical_notes'] if note]
            lexical_notes_created += len(line.lexical_notes)
            line.save()

    job.report(f"Completed stage 1: {lexical_notes_created} lexical notes created.")


def create_translations(job, gemini_client, model, thinking_config, usage_totals=None):
    corpus = job.corpus
    syriac_text = job.content
    stage = 3
    lines_translated = 0
    translator_notes_created = 0

    # grab our lexical notes prompt
    prompt = corpus.get_content('Prompt', {'name': 'Translator'}, single_result=True)

    # construct examples
    example_blocks = []
    example_lines = corpus.get_content('ExampleLine', all=True).order_by('line_number')[:10]
    for example_line in example_lines:
        example_block = f'''Input:\n<LINE number={example_line.line_number}>{example_line.original}</LINE>\n'''
        example_block += f'''Output:\n<TRANSLATION number={example_line.line_number}>{example_line.translation}</TRANSLATION>'''
        example_blocks.append(example_block)

    # grab syriac lines
    syriac_lines = corpus.get_content('SyriacLine', {'syriac_text': syriac_text.id}).order_by('line_number')

    # run them through model and get deliverables
    line_deliverables = run_lines_through_model(
        job,
        stage,
        syriac_lines,
        gemini_client,
        model,
        thinking_config,
        prompt,
        {'examples': '\n\n'.join(example_blocks)},
        usage_totals=usage_totals
    )

    for line in syriac_lines:
        if line.line_number in line_deliverables:
            deliverable_found = False

            if line_deliverables[line.line_number]['translation']:
                line.translation = line_deliverables[line.line_number]['translation']
                lines_translated += 1
                deliverable_found = True

            if line_deliverables[line.line_number]['translator_notes']:
                line.translator_notes = [tn for tn in line_deliverables[line.line_number]['translator_notes'] if tn]
                translator_notes_created += len(line.translator_notes)
                deliverable_found = True

            if deliverable_found:
                line.save()

    job.report(f"Completed stage 2: {lines_translated} lines translated and {translator_notes_created} translator notes created.")


def review_lines(job, gemini_client, model, thinking_config, usage_totals=None):
    corpus = job.corpus
    syriac_text = job.content
    stage = 4
    revisions_performed = 0
    reviewer_notes_created = 0

    # grab our lexical notes prompt
    prompt = corpus.get_content('Prompt', {'name': 'Reviewer'}, single_result=True)

    # grab syriac lines
    syriac_lines = corpus.get_content('SyriacLine', {'syriac_text': syriac_text.id}).order_by('line_number')

    # run them through model and get deliverables
    line_deliverables = run_lines_through_model(
        job,
        stage,
        syriac_lines,
        gemini_client,
        model,
        thinking_config,
        prompt,
        {},
        True,
        usage_totals=usage_totals
    )

    for line in syriac_lines:
        if line.line_number in line_deliverables:
            line_modified = False

            if line_deliverables[line.line_number]['revision']:
                line.revision = line_deliverables[line.line_number]['revision']
                line_modified = True
                revisions_performed += 1

            if line_deliverables[line.line_number]['reviewer_notes']:
                line.reviewer_notes = [note for note in line_deliverables[line.line_number]['reviewer_notes'] if note]
                line_modified = True
                reviewer_notes_created += len(line.reviewer_notes)

            if line_modified:
                line.save()

    job.report(f"Completed stage 4: {revisions_performed} revisions performed and {reviewer_notes_created} revision notes created.")


def normalize_syriac_word(word):
    word = unicodedata.normalize("NFC", word)
    word = word.translate(INVISIBLE_CHARS)
    word = SYRIAC_MARKS_RE.sub("", word)
    word = re.sub(r"[܀:؛،.]", " ", word)
    word = re.sub(r"\s+", " ", word).strip()
    return word


def gather_input(line_chunk, input_stats, gather_translations=False):
    chunk_input = ""
    heading_markers = ("ܡܐܡܪ", "ܡܐܐܡܪ", "ܕܡܪܝ", "ܕܡܝ", "ܫܠܡ")

    for line in line_chunk:
        chunk_input += f'''<LINE number="{line.line_number}">{line.original}</LINE>\n'''

        # gather any rare word related lemmas, glosses, and definitions
        if line.rare_words and len(line.rare_words) == len(line.rare_words_normalized):
            for rare_word_index in range(0, len(line.rare_words)):
                rare_word = line.rare_words[rare_word_index]
                rare_word_normalized = line.rare_words_normalized[rare_word_index]

                context_parts = [
                    f'''\n<CONTEXT number="{line.line_number}">Word: {rare_word}''',
                    f'''Normalized: {rare_word_normalized.normalized}'''
                ]

                if rare_word_normalized.lemma:
                    context_parts.append(f'''Lemma: {rare_word_normalized.lemma}''')
                if rare_word_normalized.glosses:
                    context_parts.append(f'''Glosses: {','.join(rare_word_normalized.glosses)}''')
                if rare_word_normalized.definitions:
                    context_parts.append(f'''Definitions: {','.join(rare_word_normalized.definitions)}''')

                chunk_input += '\n'.join(context_parts) + "</CONTEXT>"
                input_stats['rare_words'] += 1

        # gather any lexical/translator notes
        line_notes = []
        if line.lexical_notes:
            line_notes += line.lexical_notes
            input_stats['lexical_notes'] += len(line.lexical_notes)

        if line.translator_notes:
            line_notes += line.translator_notes
            input_stats['translator_notes'] += len(line.translator_notes)

        if line_notes:
            chunk_input += f'''<NOTES number="{line.line_number}">{'\n\n'.join(line_notes)}</NOTES>\n'''

        # gather any translations and check for review priority
        if gather_translations:
            review_priority = 'optional'
            review_reason = ''

            if not line.translation:
                review_reason = "translation missing"
                line.translation = ""

            # check if opening lines
            if line.line_number < 4:
                review_reason = "opening heading/title line"

            # check for heading markers
            if any(marker in line.original for marker in heading_markers):
                review_reason = "heading, title, numbered label, or colophon marker"

            # check for rare words
            elif line.rare_words:
                review_reason = "line has rarely occurring words"

            if review_reason:
                review_priority = 'mandatory'
                input_stats['mandatory_revisions'] += 1

            chunk_input += f'''<TRANSLATION number="{line.line_number}" review-priority="{review_priority}" review-reason="{review_reason}">{line.translation}</TRANSLATION>\n'''

        input_stats['lines'] += 1

    return chunk_input


def run_lines_through_model(job, stage, syriac_lines, gemini_client, model, thinking_config, prompt, context={}, gather_translations=False, usage_totals=None):
    input_stats = {
        'lines': 0,
        'rare_words': 0,
        'lexical_notes': 0,
        'translator_notes': 0,
        'mandatory_revisions': 0,
    }
    stage_usage = new_usage_totals()
    line_deliverables = {}
    total_lines = syriac_lines.count()

    # iterate over the lines to form our chunked API calls:
    for line_start in range(0, total_lines, LINE_CHUNK_SIZE):
        line_chunk = syriac_lines[line_start:line_start + LINE_CHUNK_SIZE]
        context['input'] = gather_input(line_chunk, input_stats, gather_translations)

        # use Django's templating engine to inject the input into the prompt
        chunk_prompt = f"{Template(prompt.template).render(Context(context))}".strip()

        if line_start == 0:
            job.report(f"Prompt for the first chunk of lines for stage {stage}:\n\n{prepare_prompt_for_report(chunk_prompt)}")

        # call the Gemini API
        model_output = None
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=chunk_prompt,
                config=thinking_config,
            )
            accumulate_usage(getattr(response, 'usage_metadata', None), stage_usage)
            model_output = response.text
        except:
            job.report(f"An error occurred while prompting {model} using {LINE_CHUNK_SIZE} lines starting at line {line_start + 1}:\n\n{traceback.format_exc()}")

        if model_output:
            if line_start == 0:
                job.report(f"Model output for the first chunk of lines for stage {stage}:\n\n{prepare_prompt_for_report(model_output)}")

            parse_model_output(model_output, line_deliverables)

        report_percent_complete(job, stage, input_stats['lines'], total_lines)

    job.report(f"Input stats for stage {stage}: \n{json.dumps(input_stats, indent=4)}")

    # roll this stage's token usage into the job-wide totals and report a per-stage cost estimate
    if usage_totals is not None:
        for key, value in stage_usage.items():
            usage_totals[key] += value
    if stage_usage['api_calls']:
        job.report(format_cost_report(model, stage_usage, f"Estimated Gemini API cost for stage {stage}"))

    return line_deliverables


def parse_model_output(output, line_deliverables):

    def ensure_data_structure(line_number):
        if line_number not in line_deliverables:
            line_deliverables[line_number] = {
                'translation': '',
                'revision': '',
                'lexical_notes': [],
                'translator_notes': [],
                'reviewer_notes': []
            }

    for number, body in LEXICAL_NOTES_RE.findall(output):
        line_number = int(number)
        ensure_data_structure(line_number)
        line_deliverables[line_number]['lexical_notes'].append(body.strip())

    for number, body in TRANSLATOR_NOTES_RE.findall(output):
        line_number = int(number)
        ensure_data_structure(line_number)
        line_deliverables[line_number]['translator_notes'].append(body.strip())

    for number, body in TRANSLATIONS_RE.findall(output):
        line_number = int(number)
        ensure_data_structure(line_number)
        line_deliverables[line_number]['translation'] = body.strip()

    for number, body in REVISIONS_RE.findall(output):
        line_number = int(number)
        ensure_data_structure(line_number)
        line_deliverables[line_number]['revision'] = body.strip()

    for number, body in reviewer_notes_RE.findall(output):
        line_number = int(number)
        ensure_data_structure(line_number)
        line_deliverables[line_number]['reviewer_notes'].append(body.strip())


def new_usage_totals():
    return {
        'api_calls': 0,
        'prompt_tokens': 0,
        'candidates_tokens': 0,
        'thoughts_tokens': 0,
    }


def accumulate_usage(usage_metadata, usage):
    # tallies token counts from a single Gemini response's usage_metadata
    usage['api_calls'] += 1
    if not usage_metadata:
        return
    usage['prompt_tokens'] += getattr(usage_metadata, 'prompt_token_count', 0) or 0
    usage['candidates_tokens'] += getattr(usage_metadata, 'candidates_token_count', 0) or 0
    usage['thoughts_tokens'] += getattr(usage_metadata, 'thoughts_token_count', 0) or 0


def format_cost_report(model, usage, label):
    # thinking tokens are billed at the output rate, so they're folded into output_tokens
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_tokens = usage['prompt_tokens']
    output_tokens = usage['candidates_tokens'] + usage['thoughts_tokens']
    input_cost = input_tokens / 1_000_000 * pricing['input']
    output_cost = output_tokens / 1_000_000 * pricing['output']
    total_cost = input_cost + output_cost
    rates_note = '' if model in MODEL_PRICING else ' (not in MODEL_PRICING - using default rates)'

    return (
        f"{label}\n"
        f"{'=' * len(label)}\n"
        f"Model: {model}{rates_note}\n"
        f"Gemini API calls: {usage['api_calls']}\n\n"
        f"Input tokens:  {input_tokens:>12,}  @ ${pricing['input']:.3f}/1M  = ${input_cost:.4f}\n"
        f"Output tokens: {output_tokens:>12,}  @ ${pricing['output']:.3f}/1M  = ${output_cost:.4f}\n"
        f"  (includes {usage['thoughts_tokens']:,} thinking tokens billed at the output rate)\n\n"
        f"Estimated cost: ${total_cost:.4f}\n\n"
        f"Pricing is approximate and set in tasks.py (MODEL_PRICING / DEFAULT_PRICING). "
        f"Free-tier quotas and tiered long-context pricing are not accounted for."
    )


def prepare_prompt_for_report(prompt):
    return prompt.replace('<', '&lt;').replace('>', '&gt;')


def report_percent_complete(job, stage, line, total_lines):
    total_stages = 4
    stage_lines = total_lines * total_stages
    lines_completed = ((stage - 1) * total_lines) + line
    job.set_status('running', percent_complete=int((lines_completed / stage_lines) * 100))
