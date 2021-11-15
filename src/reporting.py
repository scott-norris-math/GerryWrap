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
    report_directory, report_filename_prefix = cm.build_reports_directory_and_filename(chamber, directory, plan)
    data = {
        'chamber': chamber,
        'plan': plan,
        'plan_name': cm.build_plan_name(chamber, plan),
        'original_plan': str(cm.determine_original_plan(chamber)),
        'number_plans': determine_number_plans(chamber),
        'plots_directory': pl.build_plots_directory(directory, ensemble_description),
        'report_directory': report_directory,
        'url_root': 'https://storage.googleapis.com/mum_project/reports/'
    }

    rendered_tex = j2_template.render(data)
    rendered_tex_filename = f'{report_filename_prefix}.tex'
    rendered_tex_path = f'{report_directory}{rendered_tex_filename}'
    cm.save_all_text(rendered_tex, rendered_tex_path)

    compile_pdf(report_directory, rendered_tex_filename, report_filename_prefix)
    compile_pdf(report_directory, rendered_tex_filename, report_filename_prefix)


if __name__ == '__main__':
    def main() -> None:
        directory = 'G:/rob/projects/election/rob/'

        reports_directory = cm.build_reports_directory(directory)
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

        j2_template = latex_jinja_env.get_template('Stmt-on-template.tex')

        for chamber in cm.CHAMBERS:  # ['TXSN']:  # , 'USCD'
            print(f"Chamber: {chamber}")

            plans = sorted(pp.get_valid_plans(chamber, pp.build_plans_directory(directory)) - {2100}, reverse=True)
            for plan in plans:
                print(f"Plan: {plan}")
                save_report(chamber, directory, j2_template, plan)
                # break


    main()
