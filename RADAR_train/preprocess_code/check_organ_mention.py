import pandas as pd
import dashscope
import time
import concurrent.futures
import re
import random
import numpy as np


def organ_mention(patient_id, report, organ):
    retry_count = 10
    retry_interval = 0.1

    report = report.replace('\n', "")
    report = re.sub(
        r"\s{2,}",
        " ",
        report,
    )

    prompt = f"""
            Please determine whether the given CT report mentions the organ ({organ}).
            Simply answer "yes" or "no". Do not add diagnoses or summaries.

            Supplementary anatomical knowledge:
            - The large bowel includes the cecum, colon, rectum, and anal canal. The cecum includes the appendix, so information related to the appendix should also be categorized under the large intestine.
            - The small bowel includes the jejunum, and ileum.
            - The splenic vein is part of the portal venous system.
            - C1 to C7 refer to the cervical vertebrae.
            - T1 to T12 refer to the thoracic vertebrae.
            - L1 to L5 refer to the lumbar vertebrae.
            - Pleural effusion is considered a description relating to the lungs.
            
            CT report: ({report})
            
            """

    for _ in range(retry_count):
        resp = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_plus, prompt=prompt
        )
        if resp.status_code == 200:
            text = resp.output["text"]
            text = text.strip().replace("\n", "")
            try:
                assert text.lower() in ["yes", "no"]
            except AssertionError as e:
                return patient_id, report, {organ: "未处理成功：非 yes-no 回答"}

            if text.lower() == "yes":
                return patient_id, report, {organ: "yes"}
            elif text.lower() == "no":
                return patient_id, report, {organ: "no"}

        elif resp.status_code == 429:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)

        else:
            return patient_id, report, {organ: "未处理成功：" + resp.message}
    return patient_id, report, {organ: "未处理成功：" + resp.message}


if __name__ == "__main__":
    import json
    import pandas as pd
    from functools import partial

    dashscope.api_key = ""  # using your own key

    organ_dict = {
        "肾上腺": "adrenal gland",
        "主动脉": "aorta",
        "大肠": "large bowel",
        "十二指肠": "duodenum",
        "食管": "esophagus",
        "胆囊": "gallbladder",
        "心脏": "heart",
        "髂动脉": "iliac artery",
        "髂静脉": "iliac vena",
        "下腔静脉": "inferior vena cava",
        "肾": "kidney",
        "肝": "liver",
        "肺": "lung",
        "胰腺": "pancreas",
        "门静脉": "portal vein",
        "肺动脉": "pulmonary artery",
        "肋骨": "rib",
        "骶骨": "sacrum",
        "小肠": "small bowel",
        "脾": "spleen",
        "胃": "stomach",
        "气管": "trachea",
        "膀胱": "bladder",
        "颈椎": "cervical vertebrae",
        "胸椎": "thoracic vertebrae",
        "腰椎": "lumbar vertebrae"
    }
    all_organs = list(organ_dict.values())
    

    data = json.load(open('../../ckpt/merlin_report.json'))

    max_workers = 26
    start_time = time.time()

    save_path = f"./merlin_report_mention.json"
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
            # desc = v['findings']
            desc = v['report']
            split = v['split']
            
            if split != 'train':
                continue
            if not isinstance(desc, str) or not isinstance(conc, str):
                continue
            
            num += 1
            if patient_id in new_info:
                print('Continue: ', num, patient_id)
                continue
            
            try:
                organs_batch += all_organs
                patients_batch += [patient_id] * len(all_organs)
                concs_batch += [str(desc)] * len(all_organs)

                if len(patients_batch) >= max_workers:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers
                    ) as executor:
                        batch_results = list(
                            executor.map(
                                organ_mention,
                                patients_batch,
                                concs_batch,
                                organs_batch,
                            )
                        )
                    
                    for res in batch_results:
                        if res[0] not in new_info:
                            new_info[res[0]] = {'report': res[1], 'mention': res[2]}
                        else:
                            new_info[res[0]]['mention'].update(res[2])
                    if num % 2000 == 0:
                        print(f'--> Save, num = {num}')
                        json.dump(new_info, open(save_path, "w"), ensure_ascii=False, indent=4)

                    patients_batch.clear()
                    concs_batch.clear()
                    organs_batch.clear()

                    current_time = time.time()
                    duration = (current_time - start_time)

                    print(
                        f"num: {num}, pid: {patient_id}, duration: {duration:.4f}, avg-speed {duration/len(batch_results):.4f} . (workers = {max_workers}), results: {res[2]}"
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
                        organ_mention,
                        patients_batch,
                        concs_batch,
                        organs_batch,
                    )
                )

                for res in batch_results:
                    if res[0] not in new_info:
                        new_info[res[0]] = {'report': res[1], 'mention': res[2]}
                    else:
                        new_info[res[0]]['mention'].update(res[2])
        print('End of for, save json...')
        json.dump(new_info, open(save_path, "w"), ensure_ascii=False, indent=4)

    except KeyboardInterrupt:
        json.dump(new_info, open(save_path, "w"), ensure_ascii=False, indent=4)
        print("Data saved during interruption. Exiting safely.")
