import pandas as pd
import dashscope
import time
import concurrent.futures
import re
import random
import numpy as np


def extract_info(patient_id, report, organ):
    report = report.strip().replace("\n", " ")
    report = re.sub(
        r"\s{2,}",
        " ",
        report,
    )

    prompt = f"""
            From the given CT report, extract the description information related to the specific anatomy ({organ}). Please follow these guidelines:
            - Precise extraction: Extract only the description relevant to {organ} directly from the report.
            - Focus on the affected site: If the report mentions a specific part of {organ}, make sure it is included in the extracted information.
            - Concise and clear: Directly extract content from the report, avoiding unnecessary explanations or background information.
            - Formatting requirement: Provide the information in the format “{organ}: description”. Ensure that {organ} is used consistently as the prefix for the entry. Even if the organ has multiple distinct parts or bilateral characteristics, treat it as a whole and return only one comprehensive description for {organ}.
            
            Supplementary anatomical knowledge:
            - The large bowel includes the cecum, colon, rectum, and anal canal. The cecum includes the appendix, so information related to the appendix should also be categorized under the large intestine.
            - The small bowel includes the jejunum, and ileum.
            - The splenic vein is part of the portal venous system.
            - C1 to C7 refer to the cervical vertebrae.
            - T1 to T12 refer to the thoracic vertebrae.
            - L1 to L5 refer to the lumbar vertebrae.
            - Pleural effusion is considered a description relating to the lungs.
            
            CT report:
            {report}
            
            """

    retry_count = 10
    retry_interval = 0.1
    for _ in range(retry_count):
        resp = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_max,
            prompt=prompt,
        )
        if resp.status_code == 200:
            text = resp.output["text"]
            text = text.strip().replace("\n", "")
            try:
                segments = text.strip().replace("：", ":").split(":")
                assert segments[0].lower() == organ
                key, value = segments[0].lower(), ":".join(segments[1:])
                return patient_id, report, {key.strip(): value.strip()}
            except AssertionError as e:
                return patient_id, report, {organ: "未处理成功" + str(e)}
        elif resp.status_code == 429:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        else:
            return patient_id, report, {organ: "未处理成功" + resp.message}



if __name__ == "__main__":
    import json
    import pandas as pd
    from functools import partial

    dashscope.api_key = ""

    data = json.load(open('../../ckpt/merlin_report.json'))
    max_workers = 36
    start_time = time.time()
    patients_processed = 0

    organ_mention = json.load(open("./merlin_report_mention.json"))
    save_path = "./merlin_report_organ_report_v1.json"
    try:
        new_info = json.load(open(save_path))
    except FileNotFoundError:
        new_info = {}

    patients_batch = []
    concs_batch = []
    organs_batch = []
    try:
        start_time = time.time()
        num = 0
        for patient_id, v in data.items():
            conc = v['impression']
            desc = v['report']
            split = v['split']
            
            desc = desc.strip().replace("\n", " ")
            desc = re.sub(
                r"\s{2,}",
                " ",
                desc,
            )
            
            if split != 'train':
                continue
            if not isinstance(desc, str) or not isinstance(conc, str):
                continue
            
            num += 1
            
            if num % 2000 == 0:
                print(f'--> Save, num = {num}')
                json.dump(new_info, open(save_path, "w"), ensure_ascii=False, indent=4)
            
            unprocessed_organs = [
                organ
                for organ,state in organ_mention[patient_id]['mention'].items()
                if state == "yes"
                and organ not in new_info.get(patient_id, {})
            ]

            if not len(unprocessed_organs):
                continue

            if patient_id not in new_info:
                new_info[patient_id] = {
                    "report": desc,
                }

            try:
                organs_batch += unprocessed_organs
                patients_batch += [patient_id] * len(unprocessed_organs)
                concs_batch += [desc] * len(unprocessed_organs)

                if len(patients_batch) >= max_workers:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers
                    ) as executor:
                        batch_results = list(
                            executor.map(
                                extract_info, patients_batch, concs_batch, organs_batch
                            )
                        )

                    for res in batch_results:
                        new_info[res[0]].update(res[2])

                    patients_batch.clear()
                    concs_batch.clear()
                    organs_batch.clear()

                    current_time = time.time()
                    duration = (current_time - start_time)

                    print(
                        f"num: {num}, pid: {patient_id}, duration: {duration:.4f}, avg-speed {duration / len(batch_results):.4f} . (workers = {max_workers}), results: {res[2]}"
                    )
                    processed_items_num = 0 
                    start_time = time.time()
            except Exception as e:
                print('fj error: ', e)
                continue

        if patients_batch:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                batch_results = list(
                    executor.map(
                        extract_info, patients_batch, concs_batch, organs_batch
                    )
                )

                for res in batch_results:
                    new_info[res[0]].update(res[2])
        print('End of for, save json...')
        json.dump(new_info, open(save_path, "w"), ensure_ascii=False, indent=4)

    except KeyboardInterrupt:
        json.dump(new_info, open(save_path, "w"), ensure_ascii=False, indent=4)
        print("Data saved during interruption. Exiting safely.")
