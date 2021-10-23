import os
import jinja2
import subprocess

import common as cm
import proposed_plans as pp
import plotting as pl


def determine_number_plans(chamber: str) -> str:
    return {
        'USCD': "1,500,000",
        'TXHD': '1,000,000',
        'TXSN': '617,270'
    }[chamber]


def compile_pdf(directory: str, tex_filename: str, output_filename_prefix: str) -> None:
    process = subprocess.Popen([
        'latex',
        '-output-format=pdf',
        '-job-name=' + output_filename_prefix,
        tex_filename], cwd=directory)
    process.wait()


def save_report(chamber: str, directory: str, j2_template: jinja2.Template, plan: int) -> None:
    seed_description, ensemble_number = cm.get_current_ensemble(chamber)
    ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)
    data = {
        'chamber': chamber,
        'plan': plan,
        'plan_name': cm.build_plan_name(chamber, plan),
        'original_plan': str(determine_original_plan(chamber)),
        'number_plans': determine_number_plans(chamber),
        'plots_directory': pl.build_plots_directory(directory, ensemble_description)
    }

    reports_directory = build_reports_directory(directory)
    report_filename_prefix = f'report_{cm.encode_chamber_character(chamber)}{plan}'
    report_directory = f'{reports_directory}{report_filename_prefix}/'
    cm.ensure_directory_exists(report_directory)

    rendered_tex = j2_template.render(data)
    rendered_tex_filename = f'{report_filename_prefix}.tex'
    rendered_tex_path = f'{report_directory}{rendered_tex_filename}'
    cm.save_all_text(rendered_tex, rendered_tex_path)

    compile_pdf(report_directory, rendered_tex_filename, report_filename_prefix)


def build_reports_directory(directory: str) -> str:
    reports_directory = f'{directory}reports/'
    return reports_directory


if __name__ == '__main__':
    def main():
        directory = 'C:/Users/rob/projects/election/rob/'

        reports_directory = build_reports_directory(directory)
        template_directory = f'{reports_directory}template/'

        latex_jinja_env = jinja2.Environment(
            block_start_string='\\BLOCK{',
            block_end_string='}',
            variable_start_string='\\VAR{',
            variable_end_string='}',
            comment_start_string='\\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(template_directory)
        )

        j2_template = latex_jinja_env.get_template('Stmt-on-C2135-ForHouse.tex')

        chamber = 'USCD'

        plan = 2135

        save_report(chamber, directory, j2_template, plan)


    main()
