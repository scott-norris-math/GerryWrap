from bs4 import BeautifulSoup  # conda install -c conda-forge beautifulsoup4
from datetime import datetime
from distutils.dir_util import copy_tree
from email.message import EmailMessage
from ftplib import FTP_TLS, FTP
import glob
from itertools import chain
import numpy as np
import os
import pandas as pd
import requests
import re
import smtplib
import ssl
import subprocess
import sys
import time
import traceback
from typing import Iterable

import common as cm
import data_transform as dt
import plan_statistics as ps
import proposed_plans as pp

HOTMAIL_ADDRESS = ''
HOTMAIL_PASSWORD = ''
ADMIN_PHONE_NUMBER = ''

FTP_HOST = ''
FTP_PORT = 21
FTP_USERNAME = ''
FTP_PASSWORD = ''
FTP_INITIAL_DIRECTORY = ''

PROPOSED_PLANS_URL_SUFFIX = '/organization/eb7dd023-f166-4613-92ff-b6b3d5c026c6?groups=redistricting'
PROPOSED_PLANS_URL = f'https://data.capitol.texas.gov{PROPOSED_PLANS_URL_SUFFIX}'


def build_proposed_plans_html_path(output_directory: str, page: int) -> str:
    suffix = f'_{page}' if page > 1 else ''
    return f'{output_directory}proposed_plans{suffix}.html'


def build_proposed_plan_html_path(chamber: str, output_directory: str, plan: int) -> str:
    chamber_character = cm.encode_chamber(chamber)
    return f'{output_directory}proposed_plan_page_{chamber_character}{plan}.html'


def save_web_page(url: str, path: str) -> None:
    page_content = retrieve_web_page(url)
    cm.save_all_text(page_content, path)


def retrieve_web_page(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
    }
    response = requests.get(url=url, headers=headers)
    response.raise_for_status()
    page_content = response.text
    return page_content


def disable_ssl_logging() -> None:
    os.environ['SSLKEYLOGFILE'] = ""


def extract_submitter(elements: list[str]) -> str:
    last_name = elements[1]
    if len(last_name) > 1:
        return " ".join(elements[0:2]).removesuffix(",")
    else:
        return " ".join(elements[0:3]).removesuffix(",")


def extract_hrefs(anchor_text: str, path: str) -> set[str]:
    page_text = cm.read_all_text(path)
    soup = BeautifulSoup(page_text, 'html.parser')

    hrefs = set()
    anchors = soup.findAll('a', attrs={'href': re.compile(anchor_text)})
    for anchor in anchors:
        hrefs.add(anchor.get('href'))
    return hrefs


def send_via_ssl(from_email_address: str, to_email_address: str, subject: str, body: str, host: str, port: int,
                 password: str) -> None:
    email_message = EmailMessage()
    email_message['From'] = from_email_address
    email_message['To'] = to_email_address
    email_message['Subject'] = subject
    email_message.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(from_email_address, password)
        server.sendmail(from_email_address, to_email_address, email_message.as_string())


def send_from_hotmail(from_email_address: str, password: str, to_email_address: str, subject: str, body: str) -> None:
    send_via_ssl(from_email_address, to_email_address, subject, body, "smtp.live.com", 25, password)


def send_text_message(subject: str, body: str, phone_number: str) -> None:
    send_from_hotmail(HOTMAIL_ADDRESS, HOTMAIL_PASSWORD, phone_number + "@txt.att.net", subject, body)


def send_text_message_to_admin(subject: str, body: str) -> None:
    send_text_message(subject, body, ADMIN_PHONE_NUMBER)


def upload_files_ftp(host: str, port: int, username: str, password: str, initial_directory: str,
                     paths: Iterable[str]) -> None:
    client = login_ftp(host, port, username, password, initial_directory)
    for path in paths:
        filename = os.path.basename(path)
        with open(path, 'rb') as file:
            client.storbinary(f'STOR {filename}', file)

    client.close()


def list_files_ftp(host: str, port: int, username: str, password: str, initial_directory: str) -> None:
    client = login_ftp(host, port, username, password, initial_directory)
    client.retrlines('LIST')
    client.close()


def login_ftp(host: str, port: int, username: str, password: str, initial_directory: str) -> FTP:
    client = FTP_TLS(timeout=120)
    client.set_debuglevel(2)
    client.connect(host, port)
    client.login(username, password)
    client.cwd(initial_directory)
    client.prot_p()
    client.set_pasv(True)
    return client


def upload_files_winscp(paths: Iterable[str]) -> None:
    backslash = '\\'
    upload_commands = '\n'.join([f'put {x.replace("/", backslash)}' for x in paths])

    script = f"""
open ftpes://USER_ID:PASSWORD@HOST/ -certificate="CERTIFICATE"
cd DIRECTORY
dir
{upload_commands} 
exit
"""
    cm.save_all_text(script, 'upload_script.txt')

    process = subprocess.Popen(
        ['C:\\Program Files (x86)\\WinSCP\\WinSCP.com', '/ini=nul',
         '/script=upload_script.txt'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):  # replace b'' with '' for Python 2
        print(line.decode().rstrip())

    os.remove('upload_script.txt')


def format_plans(plans: set[int]) -> str:
    return ', '.join(map(lambda x: str(x), sorted(plans, reverse=True)))


def download_url(url: str, output_path: str) -> None:
    print(f"Downloading: {url} Start")
    r = requests.get(url, allow_redirects=True)
    file = open(output_path, mode='wb')
    file.write(r.content)
    file.close()
    print(f"Downloading: {url} End")


def save_proposed_plans_page(output_directory: str, page: int) -> None:
    url = PROPOSED_PLANS_URL
    if page > 1:
        url = url + f'&page={page}'
    path = build_proposed_plans_html_path(output_directory, page)
    save_web_page(url, path)
    time.sleep(10)


def replace_plans_metadata(chamber: str, root_directory: str, output_directory: str, number_pages: int) -> None:
    proposed_plans_metadata = parse_plans_metadata(chamber, output_directory, number_pages)
    update_plans_metadata(chamber, pp.build_plans_directory(root_directory), proposed_plans_metadata)


def parse_plans_metadata(chamber: str, output_directory: str, number_pages: int) -> list:
    return list(chain(*[parse_plans_metadata_from_page(chamber, output_directory, x)
                        for x in range(1, number_pages + 1)]))


def parse_plans_metadata_from_page(chamber: str, output_directory: str, page: int) -> list:
    path = build_proposed_plans_html_path(output_directory, page)

    page_text = cm.read_all_text(path)
    soup = BeautifulSoup(page_text, 'html.parser')

    plans = {}
    chamber_character = cm.encode_chamber(chamber).lower()
    for div in soup.findAll('div', attrs={'class': re.compile('dataset-content')}):
        anchors = div.findAll('a', attrs={'href': re.compile(f'/dataset/plan{chamber_character}2')})
        if len(anchors) == 1:
            anchor = anchors[0]
            plan = int(anchor.get('href').replace(f'/dataset/plan{chamber_character}', ''))
            for inner_div in div.findAll('div'):
                # SEN. CREIGHTON SD 3 & 4 CMTE AMEND PLANS2101 (SB4)
                description = inner_div.string
                elements = description.split(" ")
                submitter = extract_submitter(elements)
                previous_plan = elements[-2]
                chamber_plan_prefix = f'PLAN{cm.encode_chamber(chamber)}'
                if previous_plan.startswith(chamber_plan_prefix):
                    try:
                        previous_plan = int(previous_plan.removeprefix(chamber_plan_prefix))
                    except ValueError:
                        previous_plan = 0
                else:
                    previous_plan = 0
                plans[plan] = (description, submitter, previous_plan)

    return [{
        'plan': plan,
        'description': description,
        'submitter': submitter,
        'previous_plan': previous_plan,
        'changed_rows': 0,
        'invalid': 0
    } for plan, (description, submitter, previous_plan) in plans.items()]


def update_plans_metadata(chamber: str, plans_directory: str, new_plans_metadata: list) -> pd.DataFrame:
    plans_metadata_df = pd.DataFrame({
        'plan': pd.Series([], dtype='int'),
        'description': pd.Series([], dtype='str'),
        'submitter': pd.Series([], dtype='str'),
        'previous_plan': pd.Series([], dtype='int'),
        'changed_rows': pd.Series([], dtype='int'),
        'invalid': pd.Series([], dtype='int')})

    # allow adding columns while keeping existing data
    if os.path.exists(pp.build_plans_metadata_path(chamber, plans_directory)):
        plans_metadata_df = pd.concat([plans_metadata_df, pp.load_plans_metadata(chamber, plans_directory)])

    for x in new_plans_metadata:
        plans_metadata_df = plans_metadata_df.append(x, ignore_index=True)

    plans_metadata_df.sort_values(by=['plan'], inplace=True)
    plans_metadata_df.set_index('plan', drop=False, inplace=True)

    pp.save_plans_metadata(chamber, plans_directory, plans_metadata_df)

    return plans_metadata_df


def parse_number_pages(output_directory: str) -> int:
    search_string = PROPOSED_PLANS_URL_SUFFIX.replace("?", "\\?")
    href_prefix = f'{search_string}&page='
    hrefs = extract_hrefs(f'^{href_prefix}', build_proposed_plans_html_path(output_directory, 1))
    page_numbers = [int(x.replace(f'{PROPOSED_PLANS_URL_SUFFIX}&page=', "")) for x in hrefs]
    return max(page_numbers)


def download_plan_raw(chamber: str, output_directory: str, plans_raw_directory: str, plan: int) -> None:
    print(f"Downloading: {plan}")
    chamber_character = cm.encode_chamber(chamber).lower()
    plan_path = f'{output_directory}plan{chamber_character}{plan}.html'
    save_web_page(f'https://data.capitol.texas.gov/dataset/plan{chamber_character}{plan}', plan_path)
    plan_download_urls = {x for x in extract_hrefs(f'plan{chamber_character}{plan}_blk.zip', plan_path)}
    if not len(plan_download_urls) == 1:
        error_string = f"Not one url for plan {plan} urls: ({','.join(plan_download_urls)})"
        raise RuntimeError(error_string)
    plan_download_url = list(plan_download_urls)[0]
    plan_blk_zip_path = f'{plans_raw_directory}plan{chamber_character}{plan}_blk.zip'
    download_url(plan_download_url, plan_blk_zip_path)
    cm.unzip_file(plans_raw_directory, plan_blk_zip_path)
    print("Download Sleep")
    time.sleep(30)


def process_proposed_plans_pages(root_directory: str, output_directory: str, plans_raw_directory: str,
                                 ensemble_statistics: dict[str, tuple[np.ndarray, np.ndarray]], file_prefix: str,
                                 known_plans: dict[str, set[int]], do_send_text_message: bool) -> dict[str, set[int]]:
    save_proposed_plans_page(output_directory, 1)
    number_pages = parse_number_pages(output_directory)
    for x in range(2, number_pages + 1):
        save_proposed_plans_page(output_directory, x)

    for chamber, chamber_known_plans in known_plans.items():
        print(f"Chamber: {chamber}")
        proposed_plans_metadata = parse_plans_metadata(chamber, output_directory, number_pages)
        proposed_plans = {x['plan'] for x in proposed_plans_metadata}

        new_plans = proposed_plans.difference(chamber_known_plans)
        if len(new_plans) > 0:
            print(f"New Plan Found: {new_plans}")

            new_plans_metadata = [x for x in proposed_plans_metadata if x['plan'] in new_plans]
            plans_directory = pp.build_plans_directory(root_directory)
            plans_metadata_df = update_plans_metadata(chamber, plans_directory, new_plans_metadata)

            if do_send_text_message:
                send_text_message_to_admin("New Plan Found", format_plans(new_plans))

            for plan in new_plans:
                download_plan_raw(chamber, output_directory, plans_raw_directory, plan)

            pp.save_current_merged_plans(chamber, root_directory, plans_metadata_df)

            print(f"Updating Plan Vectors - Start")
            pp.save_current_plan_vectors(chamber, root_directory)
            if chamber == 'USCD':
                pp.save_current_vra_plan_vectors(chamber, root_directory)
            print(f"Updating Plan Vectors - End")

            known_plans[chamber] = proposed_plans

            valid_proposed_plans = pp.determine_valid_plans(plans_metadata_df)
            ps.save_statistics_statements(chamber, root_directory, ensemble_statistics, file_prefix,
                                          valid_proposed_plans)

            pp.save_graph_filtered(chamber, root_directory, plans_metadata_df)
            graph_path = pp.build_graph_path(chamber, root_directory)
            upload_files_winscp([graph_path])

            copy_tree(f'{root_directory}plan_vectors/', 'SYNC_DIRECTORY', update=True)

    return known_plans


def handle_exception(exc_type, exc_value, exc_traceback):
    """ handle all exceptions """

    # KeyboardInterrupt is a special case.
    # We don't raise the error dialog when it occurs.
    if issubclass(exc_type, KeyboardInterrupt):
        return

    filename, line, _, _ = traceback.extract_tb(exc_traceback).pop()
    filename = os.path.basename(filename)
    error = "%s: %s" % (exc_type.__name__, exc_value)

    #send_text_message_to_admin(f"A critical error has occurred - {error}", f"line {line} of file {filename}.")

    print("Closed due to an error. This is the full error report:")
    print()
    print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

    sys.exit(1)


def get_downloaded_plans(chamber: str, plans_raw_directory: str) -> set[int]:
    plans = set()
    for path in glob.glob(f'{plans_raw_directory}plan{cm.encode_chamber(chamber)}*_blk.zip'):
        path = os.path.normpath(path)
        plan_string = str(path.replace(os.path.dirname(path), '').removesuffix('_blk.zip')[1:]). \
            removeprefix(f'plan{cm.encode_chamber(chamber).lower()}')
        plans.add(int(plan_string))
    return plans


def run(root_directory: str, output_directory: str, do_send: bool) -> None:
    sys.excepthook = handle_exception

    plans_raw_directory = pp.build_plans_raw_directory(root_directory)

    print("Loading Ensembles Statistics")
    file_prefix = dt.build_election_filename_prefix('PRES20', 'votes')
    ensemble_statistics = {chamber: ps.load_ensemble_statistics(chamber, root_directory, file_prefix) for chamber in
                           cm.CHAMBERS}

    known_plans = {chamber: get_downloaded_plans(chamber, plans_raw_directory) for chamber in cm.CHAMBERS}
    for (chamber, plans) in known_plans.items():
        print(f"Initial Plans: {chamber} {format_plans(plans)}")

    while True:
        print(datetime.now().strftime("%H:%M:%S"))

        known_plans = process_proposed_plans_pages(root_directory, output_directory, plans_raw_directory,
                                                   ensemble_statistics, file_prefix, known_plans,
                                                   do_send)
        print("Sleeping")
        time.sleep(10 * 60)


if __name__ == '__main__':
    def main() -> None:
        print("start")

        # Have to set the key log file to nothing or else we get a permission error
        disable_ssl_logging()

        root_directory = 'G:/rob/projects/election/rob/'
        output_directory = 'scrape_data/'
        cm.ensure_directory_exists(output_directory)
        do_send = False

        if False:
            for x in cm.CHAMBERS:
                # update_plans_metadata(x, pp.build_plans_directory(root_directory), [])
                replace_plans_metadata(x, root_directory, output_directory, 5)

        if False:
            graph_path = pp.build_graph_path('USCD', root_directory)
            upload_files_winscp([graph_path])

        if True:
            run(root_directory, output_directory, do_send)

        if False:
            print(parse_number_pages(output_directory))

        print("end")


    main()
