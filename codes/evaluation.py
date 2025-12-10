"""
Medical Diagnosis Evaluation Script

This script automates the evaluation of medical diagnosis dialogues using LLMs.
"""
import json
import openai
import jsonlines
import tqdm

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed # 引入并行计算库


class MedicalEvaluator:
    """
    Evaluator class for assessing medical consultations.
    Handles API communication, prompt construction, and score calculation.
    """
    
    # --- 修改部分: 在初始化时接收并行工作线程数 ---
    def __init__(self, model: str = "gpt-oss-120b", base_url: str = None, api_key: str = None, temperature: float = 0.0, max_workers: int = 4):
        """
        Initialize the evaluator.
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
        # self.lock = threading.Lock() # Lock not required in current design as main thread collects results

    def chat(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and retrieves the response.
        
        Args:
            prompt (str): The input prompt for the model.
            
        Returns:
            str: The content of the model's response.
        """
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional medical evaluation agent. Strictly follow the instructions to complete your tasks."},
                    {"role": "user", "content": '/no_think\n' + prompt}
                ],
                temperature=self.temperature,
                max_tokens=100000,
            )
            
        response_content = response.choices[0].message.content.strip()
        return response_content
    
    def call_openai_api(self, case_data):
        """
        Core evaluation logic. specific prompts are constructed for different metrics,
        sent to the API, and results are parsed.
        
        Args:
            case_data (dict): The dictionary containing medical case details and dialogue history.
            
        Returns:
            dict: A dictionary containing the raw evaluation metrics and explanations.
        """
        if True:
            # ---------------------------------------------------------
            # 1. Extract Communication Record
            # ---------------------------------------------------------
            try:
              # Parse Registration dialogue
              reg_communiaction = []
              for i in range(len(case_data['doctor_patient_diaglogues'][0]['diaglogue'])):
                reg_communiaction.append('doctor: "'+case_data['doctor_patient_diaglogues'][0]['diaglogue'][i]['doctor'].split('```json')[0].strip()+'"')
                if 'patient' in case_data['doctor_patient_diaglogues'][0]['diaglogue'][i].keys():
                  reg_communiaction.append('patient: "'+case_data['doctor_patient_diaglogues'][0]['diaglogue'][i]['patient'].strip()+'"')
              reg_communiaction = '[Registration]\n' + '\n'.join(reg_communiaction)
              
              # Parse Specialist Consultation dialogue
              diag = case_data['doctor_patient_diaglogues'][1]
              consul_communiaction = []
              for i in range(len(diag['diaglogue'])):
                consul_communiaction.append('doctor: "'+diag['diaglogue'][i]['doctor'].split('```json')[0].strip()+'"')
                if 'patient' in diag['diaglogue'][i].keys():
                  consul_communiaction.append('patient: "'+diag['diaglogue'][i]['patient'].strip()+'"')
              consul_communiaction = reg_communiaction + '[Specialist Consulation]\n' + '\n'.join(consul_communiaction)
            except:
              consul_communiaction = ''

            # ---------------------------------------------------------
            # 2. Final Diagnosis Accuracy (FDA) & Recall
            # ---------------------------------------------------------
            try:
              diagnosis = case_data['patient_records'][-1]['final_diagnosis'][:5]
              pred = diagnosis[0]
              gt = case_data['original_data']['Final Diagnosis']

              # FDA_prompt: Evaluates if the student's diagnosis matches the ground truth.
              # Scoring: 2=Exact Match, 1=Category Match, 0=Incorrect.
              FDA_prompt = f"You are an expert in diagnosing challenging cases. You will receive a student's answer containing 5 differential diagnoses, as well as the reference diagnosis. You need to score each diagnosis from the student's answer according to the following rules: 2 = The student's diagnosis exactly matches the reference diagnosis; 1 = A broader or narrower disease category that includes or is included in the reference diagnosis (e.g., “pneumonia” vs “bacterial pneumonia”).; 0 = Unrelated or incorrect. Here is the student's answer: {','.join(diagnosis)}. Here is the reference diagnosis: {gt}. Output Format: Output the scores in the following format. 1. Disease 1 Name: <The Score of Disease 1>; 2. Disease 2 name: <The Score of Disease 2>; ..."

              # fda_output = self.chat(FDA_prompt).split('@')[1:]
              # fda_3 = fda_output[-1].split(':')[-1].strip()
              # fda_1 = fda_output[0].split(':')[-1].strip()
              
              # Parse scores to determine Top-1, Top-3, and Top-5 accuracy
              fda_output = self.chat(FDA_prompt).split(';')
              scores = []
              fda_1 = 'No'
              fda_3 = 'No'
              fda_5 = 'No'
              for s in fda_output:
                s = s.split(':')[-1].strip().strip('.')
                # print(s)
                scores.append(s)
              if scores[0]=='2':
                fda_1 = 'Yes'
              if '2' in scores[:3]:
                fda_3 = 'Yes'
              if '2' in scores:
                fda_5 = 'Yes'
            except:
              fda_1 = 'No'
              fda_3 = 'No'
              fda_5 = 'No'
            
            # ---------------------------------------------------------
            # 3. Medical Test Utilization (IoU & Efficiency)
            # ---------------------------------------------------------
            try:
              original_tests = case_data['original_data']['Diagnostic Tests']
              suggested_tests = []
              for r in case_data['patient_records']:
                if r['Recording Department'] == 'diagnostic_test':
                  if r['Completed Tests']=="[]":
                    continue
                  tests = r['Completed Tests'].strip('[]').split(',')
                  for t in tests:
                    suggested_tests.append(t.strip().strip("'"))
              suggested_tests = list(set(suggested_tests))
              
              # Prompt to normalize and compare suggested tests vs reference tests
              test_check_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given the reference medical tests: **{original_tests}** and the doctor\'s suggested medical tests: **{",".join(suggested_tests)}**, perform the following steps: 1. Deduplicate the tests and merge items that refer to the same test. 2. Identify which suggested tests are equivalent to the reference tests (same clinical purpose, even if named differently). Output format strictly: @ Reference tests: item1, item2, ... @ Suggested tests in practice: item1, item2, ... @ Equivalent tests: item1, item2, ...\n- Use consistent terminology. - Only output in the above format.- Do not include extra text.'
              tests_check_explanation = self.chat(test_check_prompt)

              # Parse results to calculate IoU (Intersection over Union)
              tests_check = tests_check_explanation.split('@')
              ref_tests = tests_check[1].split(':')[-1].strip().split(',')
              ref_tests = [t.strip() for t in ref_tests if t.strip().lower()!='none' and len(t.strip())>1]
              suggest_tests = tests_check[2].split(':')[-1].strip().split(',')
              suggest_tests = [t.strip() for t in suggest_tests if t.strip().lower()!='none' and len(t.strip())>1]
              eq_tests = tests_check[3].split(':')[-1].strip().split(',')
              eq_tests = [t.strip() for t in eq_tests if t.strip().lower()!='none' and len(t.strip())>1]
              test_iou = len(eq_tests)/(len(ref_tests)+len(suggest_tests)-len(eq_tests)) if (len(ref_tests)+len(suggest_tests)-len(eq_tests))!=0 else 1

              # Check tool usage for retrieved tests
              re_tests = [] #retrieved tests
              if len(case_data['tool_usages'])!=0:
                for t in case_data['tool_usages']:
                  if t['name']!='get_info':
                    continue
                  for tn in t['args']['test_names']:
                    re_tests.append(tn) # retrieved tests 
              re_tests = list(set(re_tests))
              retrieve_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given the doctor\'s suggested medical tests: **{suggest_tests}** and the retrieved tests from knowledge tools: **{",".join(re_tests)}**, identify which suggested tests are equivalent to the retrieved tests (same clinical purpose). Output format strictly: @ Retrieved Suggested tests: item1, item2, ...\n- Use consistent terminology. - Only output in the above format. - Do not include extra text.'
              re_tests = self.chat(retrieve_prompt).split(':')[-1].strip().split(',')

            except:
              test_iou = 0.0
              re_tests = []
              suggest_tests = [None]
              eq_tests = []
              ref_tests = []
              tests_check_explanation = None

            # ---------------------------------------------------------
            # 4. Department Allocation Accuracy
            # ---------------------------------------------------------
            try:
              case_info = case_data['original_data']['Case Information'] + '[Physical Exam]' + case_data['original_data']['Physical Examination']
              right_diagnosis = case_data['original_data']['Final Diagnosis']
              department_suggestion = case_data['patient_records'][1]['suggestion']
              reason = case_data['patient_records'][1]['clinic_note']
              prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given the medical case: **{case_info}** and the correct diagnosis: **{right_diagnosis}**, the doctor in registration reception leads the patient to the **{department_suggestion}** department of the hospital and gives a reason: **{reason}**. Do you think this initial department suggestion is correct? Score the suggestion based on following rules: Scoring standard: 2 = The suggested department fully matches the correct diagnosis pathway and is clearly the most appropriate destination (e.g., chest pain → Cardiology for myocardial infarction). 1 = The suggested department is somewhat reasonable but not optimal (e.g., dizziness → Neurology, but should ideally go to ENT based on final diagnosis). 0 = The suggested department is completely inappropriate or inconsistent with the diagnosis (e.g., appendicitis → Dermatology). Only response the score <0 or 1 or 2>, and no other content is allowed.'
              allocation_accuracy = self.chat(prompt)
              if '2' in allocation_accuracy:
                allocation_accuracy = 'Yes'
            except:
              allocation_accuracy = 'No'
            
            # ---------------------------------------------------------
            # 5. Tool Call Analysis (Accuracy & Statistics)
            # ---------------------------------------------------------
            try:
              total_calls = len(case_data['tool_usages'])
              valid_calls = 0
              prof_tool_calls = 0
              mdt_calls = 0
              prof_tool_stat = {'organize_mdt':0, 'search_pubmed':0, 'search_wikipedia':0}
              for c in case_data['tool_usages']:
                if c['name'] not in ['get_info', 'organize_mdt', 'search_pubmed', 'search_wikipedia']:
                  continue
                elif len(c['args'])>0:
                  valid_calls += 1
                  if c['name'] in ['organize_mdt', 'search_pubmed', 'search_wikipedia']:
                    prof_tool_calls += 1
                    prof_tool_stat[c['name']] += 1
              tca = valid_calls/total_calls if total_calls!=0 else 1
            except:
              total_calls = None
              valid_calls = None
              prof_tool_calls = 0
              prof_tool_stat = {'organize_mdt':0, 'search_pubmed':0, 'search_wikipedia':0}
              tca = 0

            # ---------------------------------------------------------
            # 6. Core Information Inquiry Completeness
            # ---------------------------------------------------------
            try:
              indentify_info_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given a medical case: **{case_info}**, and its confirmed final diagnosis: **{right_diagnosis}**, perform the following tasks: 1. List all the core and most important information a doctor must ask the patient before making a correct diagnosis (e.g., past medical history, family history). 2. From the actual doctor communication record: **{consul_communiaction}**, identify which of the core information you mentioned has actually been asked by the doctor. Only consider information that is a direct match to your core list, only include items that are present in your core information list. Output format: @ Core information needed: info 1, info 2, ... \n@ Inquired information in practice: info 1, info 2, ... \n- No additional text, explanation, or punctuation is allowed. - Use consistent terminology to ensure exact matching.'
              indentified_info = self.chat(indentify_info_prompt)
              indentified_infos = indentified_info.split('@')
              required_infos = indentified_infos[1].split(':')[-1].split(',')
              required_infos = [i.strip() for i in required_infos]
              inquired_infos = indentified_infos[2].split(':')[-1].split(',')
              inquired_infos = [i.strip() for i in inquired_infos]
            except:
              required_infos = [None]
              inquired_infos = []
            
            # ---------------------------------------------------------
            # 7. MDT (Multi-Disciplinary Team) Requirement Identification
            # ---------------------------------------------------------
            try:
              mdt_neccessary_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given a medical case: **{case_info}**, and its corresponding correct final diagnosis: **{right_diagnosis}**, you need to identify whether Multi-disciplinary teamwork (MDT) is very necessary to solve this case. Evaluate MDT necessity based on the following criteria: 1. Multi-system involvement: Are ≥2 organ systems affected? 2. Diagnostic difficulty: Are symptoms atypical or differential diagnoses multiple? 3. Treatment complexity: Does therapy require multiple specialties (surgery, chemo, rehab, psych)? 4. Risk or safety: High perioperative risk, drug interactions, or complex care coordination needed? Output format: <Yes/No. Brief explanation within 2 sentences>'
              mdt_neccessary_explanation = self.chat(mdt_neccessary_prompt)
              mdt_neccessary = mdt_neccessary_explanation.split('.')[0].strip().lower()
              if 'yes' in mdt_neccessary:
                if mdt_calls > 0:
                  mdt_identified = True
                else:
                  mdt_practice_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. We have confirmed that a medical case needs Multi-disciplinary teamwork (MDT). You need to identify whether a doctor successfully identify this need too. Here is the communication records of the doctor: **{consul_communiaction}**. '
                  if prof_tool_calls>0:
                    tools = [c['name']+':'+c['args']['query'] for c in case_data['tool_usages'] if c['name'] in ['search_pubmed', 'search_wikipedia']]
                    mdt_practice_prompt += f'The doctor also calls professional tools, and here is the records: **{";".join(tools)}**. '
                  mdt_practice_prompt += 'Output only Yes or No, no other contents is allowed.'
                  mdt_identified = 'yes' in self.chat(mdt_practice_prompt).lower()
              else:
                mdt_identified = True
            except:
              mdt_identified = False

            # ---------------------------------------------------------
            # 8. Clinical Logic Coherence (CLC) - Subjective Metric
            # ---------------------------------------------------------
            try:
              logic_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Based on the doctor consultation record: **{consul_communiaction}** and the reference correct diagnosis: **{right_diagnosis}**, you need to identify the Clinical Logic Coherence of doctor. Standard: - 10 points: Complete and coherent clinical reasoning chain — "inquiry → tests → diagnosis → treatment" — with consistent causal logic (e.g., inquiry about dysphagia → esophageal test → diagnosis of esophageal injury → mucosal protective drugs). - 5 points: Partially coherent reasoning; some steps logically related but missing one link or minor mismatch. - 0 points: Major logical break or inconsistency (e.g., inquiry about dysphagia → no esophageal test → unrelated diagnosis). Notes: - Do not assume missing steps; rely only on the given record. - Planned but not yet performed steps count if explicitly reasoned. - Apply strict and responsible judgment; avoid 0 or 10 unless strongly justified. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              clc_output = self.chat(logic_prompt)
              clc_score = float(clc_output.split('@')[1].split(':')[1].strip('*').strip())
              
              clc_reason = clc_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              clc_score = 0.0
              clc_reason = None

            # ---------------------------------------------------------
            # 9. Privacy Invasion Prevention (PIP)
            # ---------------------------------------------------------
            try:
              privacy_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Based on the doctor consultation record: **{consul_communiaction}**, evaluate the degree to which the doctor adheres to Privacy Invasion Prevention (PIP). Scoring standard: - 10 points: Inquiries are strictly limited to condition-related information, with no irrelevant personal questions (e.g., income, relationship status). - 5 points: Generally appropriate but includes one or two mildly irrelevant or borderline personal questions without persistent intrusion. - 0 points: Repeated or explicit inquiries about irrelevant personal details (e.g., "What is your monthly income?"). Notes: - Consider whether questions are medically relevant (e.g., marital status for reproductive context is acceptable). - Consider whether the question was initiated by the patient or the doctor. - Apply strict and responsible judgment when scoring, and be careful when assigning 0 or 10. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              pip_output = self.chat(privacy_prompt)
              pip_score = float(pip_output.split('@')[1].split(':')[1].strip('*').strip())
              pip_reason = pip_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              pip_score = 0.0
              pip_reason = None

            # ---------------------------------------------------------
            # 10. Transparency of Medical Explanations (TME)
            # ---------------------------------------------------------
            try:
              transparency_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Based on the doctor consultation record: **{consul_communiaction}**, evaluate the degree to which the doctor adheres to Transparency of Medical Explanations (TME). Scoring standard: - 10 points: The doctor actively explains the reason for questions, the purpose of tests, and the diagnostic basis (e.g., "A CT scan is needed to check for airway swelling"). - 5 points: Some explanations are provided but lack completeness or clarity about diagnostic purpose. - 0 points: The doctor provides no explanations, using only mechanical questions or instructions. Notes: - Consider the overall communication context. - Implicit explanations (e.g., "Let\'s check if there\'s inflammation") count as partial transparency. - Apply strict and responsible judgment when scoring, be careful when scoring 0 or 10. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              tme_output = self.chat(transparency_prompt)
              tme_score = float(tme_output.split('@')[1].split(':')[1].strip('*').strip())
              tme_reason = tme_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              tme_score = 0.0
              tme_reason = None

            # ---------------------------------------------------------
            # 11. Individualized Treatment Recommendation Rationality (ITRR)
            # ---------------------------------------------------------
            try:
              doctor_final = diag['diaglogue'][-1]['doctor']
              individual_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given a medical case: **{case_info}**, corresponding tests: **{original_tests}** and the correct diagnosis reference: **{right_diagnosis}**, you need to identify whether the doctor\'s treatment recommendation: **{doctor_final}**, is rational and suitable according to the patient\'s individual situation. Standard: - 10 points: Based on symptoms/tests, adjusted for comorbidities/age, covers core interventions, complies with guidelines. - 5 points: Generally evidence-based but lacks full individual adjustment or omits one key element. - 0 points: No evidence support, conflicts with contraindications, or incomplete plan. Rules: - Use only the provided information (do not infer missing data). - Apply strict and responsible judgment; avoid 0 or 10 unless clearly justified.Apply strict and responsible judgment when scoring, be careful when scoring 0 or 10. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              itrr_output = self.chat(individual_prompt)
              itrr_score = float(itrr_output.split('@')[1].split(':')[1].strip('*').strip())
              itrr_reason = itrr_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              itrr_score = 0.0
              itrr_reason = None

            # ---------------------------------------------------------
            # 12. Medical Record Compliance (MRC)
            # ---------------------------------------------------------
            try:
              record = ''
              for r in case_data['patient_records']:
                if 'clinic_note' in r.keys():
                  stage = r['Recording Department']
                  record += f'[{stage}]\n' + r['clinic_note'] + '\n'
              record = record.strip()

              record_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Evaluate the compliance of the doctor\'s clinical note (**{record}**) based on the doctor-patient communication record (**{consul_communiaction}**). You need to Evaluate whether the clinical note accurately reflects patient-reported symptoms and history, is well-organized, and uses precise medical terminology. Scoring standard: - 10 points: Fully cites patient-reported information; structure is clear; terminology is precise and professional (e.g., “stridor” instead of “throat noise”). - 5 points: Captures part of the patient\'s report but lacks completeness or uses some non-standard wording. - 0 points: Misses key patient information, disorganized, or vague (e.g., “Patient unwell, order tests”). Apply strict and responsible judgment when scoring, be careful when scoring 0 or 10. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              mrc_output = self.chat(record_prompt)
              mrc_score = float(mrc_output.split('@')[1].split(':')[1].strip('*').strip())
              mrc_reason = mrc_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              mrc_score = 0.0
              mrc_reason = None

            # ---------------------------------------------------------
            # 13. Physician Patient Communication Caring Effectiveness (PPCCE)
            # ---------------------------------------------------------
            try:
              caring_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given the communication record between doctor and patient: **{consul_communiaction}**, you need to Assess the **Caring Effectiveness** of the physician\'s communication **from the patient\'s perspective**. Scoring standard: - 10 points: Expresses empathy, caring, and respect (e.g., “I understand that must be worrying for you.”). - 5 ponits: Polite and professional but emotionally neutral (e.g., “Please take the medication as prescribed.”). - 0 points: Rude, dismissive, or lacking empathy (e.g., “Stop wasting my time.”). Apply strict and responsible judgment when scoring, be careful when scoring 0 or 10. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              ppcce_output = self.chat(caring_prompt)
              ppcce_score = float(ppcce_output.split('@')[1].split(':')[1].strip('*').strip())
              ppcce_reason = ppcce_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              ppcce_score = 0.0
              ppcce_reason = None

            # ---------------------------------------------------------
            # 14. Follow up Prognosis Management (FPM)
            # ---------------------------------------------------------
            try:
              treatment_plan = case_data['patient_records'][-1]['treatment_plan']
              fpm_prompt = f'You are an experienced clinical expert familiar with medical diagnoses. Given a medical case: **{case_info}**, corresponding tests: **{original_tests}** and the correct diagnosis reference: **{right_diagnosis}**, you need to evaluate whether the doctor well conduct the Follow up Prognosis Management according to the treatment plan suggestion: **{treatment_plan}**. Standard: - 10 points: Clearly includes follow-up timing, recheck/test items, and patient guidance (e.g., “Recheck barium swallow in 2 months; avoid irritant foods”). -5 points:  Mentions follow-up or recheck but lacks completeness (missing one or two components). - 0 points: No follow-up/guidance (e.g., only "Treatment completed"). Apply strict and responsible judgment when scoring, be careful when scoring 0 or 10. Output in the following format: @Rating: <integer score 0-10> \n@Reason: <one-sentence explanation of the rating>'
              fpm_output = self.chat(fpm_prompt)
              fpm_score = float(fpm_output.split('@')[1].split(':')[1].strip('*').strip())
              fpm_reason = fpm_output.split('@')[2].split(':')[1].strip('*').strip()
            except:
              fpm_score = 0.0
              fpm_reason = None

            # ---------------------------------------------------------
            # Assemble Final Output Dictionary
            # ---------------------------------------------------------
            output_json = {
  "case_id": case_data['original_data']['id'],
  "objective_metrics": {
    "Work_Stream_Completion_WSC": {
      "process_success": case_data['process_success']
      },
    "Final_Diagnosis_Accuracy_FDA": {
      "diagnosis_correct": fda_1.lower()=='yes',
    },
    "Diagnosis_Recall@3_DR@3": {
      "diagnosis_correct": fda_3.lower()=='yes',
    },
    "Diagnosis_Recall@5_DR@5": {
      "diagnosis_correct": fda_5.lower()=='yes',
    },
    "IoU_of_Medical_Test_IoU@MT": {
      "IoU": test_iou,
      "explanation": tests_check_explanation
    },
    "Test_Result_Utilization_Rate_TRUR": {
      "utilized_test_nums": len(re_tests),
      "required_test_nums": len(suggest_tests),
      "rate": min(len(re_tests)/len(suggest_tests), 1.0) if len(suggest_tests)!=0 else 1,
    },
    "Department_Allocation_Accuracy_DAA": {
      "allocation_correct": allocation_accuracy.lower()=='yes',
    },
    "Tool_Call_Accuracy_TCA": {
      "valid_tool_call_nums": valid_calls,
      "total_tool_call_nums": total_calls,
      "accuracy": tca,
      "organize_mdt":prof_tool_stat['organize_mdt'],
      "search_pubmed":prof_tool_stat['search_pubmed'],
      "search_wikipedia":prof_tool_stat['search_wikipedia'],
    },
    "Core_Information_Inquiry_Completeness_CIIC": {
      "inquired_info_nums": len(set(inquired_infos)),
      "required_info_nums": len(set(required_infos)),
      "inquiry_completeness": min(len(inquired_infos)/len(required_infos), 1.0) if len(required_infos)!=0 else 1, 
    },
    "Correct_Tool_Calls_CTC": {
      "correct_call_nums": prof_tool_calls
    },
    "MDT_Requirement_Identification_Accuracy_MDTRA": {
      "mdt_identified": mdt_identified,
    }
  },
  "subjective_metrics": {
    "Clinical_Logic_Coherence_CLC": {
      "score": clc_score,
      "explanation": clc_reason
    },
    "Privacy_Invasion_Prevention_PIP": {
      "score": pip_score,
      "explanation": pip_reason
    },
    "Transparency_of_Medical_Explanations_TME": {
      "score": tme_score,
      "explanation": tme_reason
    },
    "Individualized_Treatment_Recommendation_Rationality_ITRR": {
      "score": itrr_score,
      "explanation": itrr_reason
    },
    "Medical_Record_Compliance_MRC": {
      "score": mrc_score,
      "explanation": mrc_reason
    },
    "Physician_Patient_Communication_Caring_Effectiveness_PPCCE": {
      "score": ppcce_score,
      "explanation": ppcce_reason
    },
    "Follow_up_Prognosis_Management_FPM": {
      "score": fpm_score,
      "explanation": fpm_reason
    }
  }
}
            
            evaluation_result = output_json
            return evaluation_result
            
        # except Exception as e:
        #     print(f"错误: OpenAI API调用失败: {e}")
        #     return None
        #     sys.exit(1)

    def calculate_scores(self, evaluation_result: Dict) -> Dict[str, float]:
        """
        Calculate numerical scores for each dimension based on the evaluation result.
        
        Args:
            evaluation_result (Dict): The dictionary returned by call_openai_api.
            
        Returns:
            Dict[str, float]: A dictionary mapping metric names to their normalized scores (0.0 - 1.0).
        """
        scores = {}
        
        objective_metrics = evaluation_result.get("objective_metrics", {})
        
        # WSC: Work Stream Completion (布尔值转为0/1)
        wsc = objective_metrics.get("Work_Stream_Completion_WSC", {})
        scores["WC"] = 1.0 if wsc.get("process_success", True) else 0.0
        
        # FDA: Final Diagnosis Accuracy
        fda = objective_metrics.get("Final_Diagnosis_Accuracy_FDA", {})
        scores["DR"] = 1.0 if fda.get("diagnosis_correct", True) else 0.0
        
        # DR@3: Diagnosis Recall@3
        dr3 = objective_metrics.get("Diagnosis_Recall@3_DR@3", {})
        scores["DR@3"] = 1.0 if dr3.get("diagnosis_correct", True) else 0.0

        # DR@5: Diagnosis Recall@5
        dr5 = objective_metrics.get("Diagnosis_Recall@5_DR@5", {})
        scores["DR@5"] = 1.0 if dr5.get("diagnosis_correct", True) else 0.0
        
        # IoU@MT: IoU of Medical Test
        iou_mt = objective_metrics.get("IoU_of_Medical_Test_IoU@MT", {})
        scores["IC"] = float(iou_mt.get("IoU", 0.0))
        
        # TRUR: Test Result Utilization Rate
        trur = objective_metrics.get("Test_Result_Utilization_Rate_TRUR", {})
        scores["RU"] = float(trur.get("rate", 0))
        
        # DAA: Department Allocation Accuracy
        daa = objective_metrics.get("Department_Allocation_Accuracy_DAA", {})
        scores["TP"] = 1.0 if daa.get("allocation_correct", True) else 0.0
        
        # TCA: Tool Call Accuracy
        tca = objective_metrics.get("Tool_Call_Accuracy_TCA", {})
        scores["TCA"] = float(tca.get("accuracy", 0))
        scores["mdt"] = float(tca.get("organize_mdt", 0))
        scores["pubmed"] = float(tca.get("search_pubmed", 0))
        scores["wikipedia"] = float(tca.get("search_wikipedia", 0))
        
        # CIIC: Core Information Inquiry Completeness
        ciic = objective_metrics.get("Core_Information_Inquiry_Completeness_CIIC", {})
        scores["IS"] = float(ciic.get("inquiry_completeness", 0.0))
        
        # CTC: Correct Tool Calls
        ctc = objective_metrics.get("Correct_Tool_Calls_CTC", {})
        scores["CTC"] = float(ctc.get("correct_call_nums", 0))
        
        # MDTRA: MDT Requirement Identification Accuracy
        mdtra = objective_metrics.get("MDT_Requirement_Identification_Accuracy_MDTRA", {})
        scores["MDTRA"] = 1.0 if mdtra.get("mdt_identified", True) else 0.0
        
        # 主观指标评分 (0-10分转为0-1分)
        subjective_metrics = evaluation_result.get("subjective_metrics", {})
        
        subjective_keys = [
            ("LC", "Clinical_Logic_Coherence_CLC"),
            ("PS", "Privacy_Invasion_Prevention_PIP"), 
            ("TME", "Transparency_of_Medical_Explanations_TME"),
            ("TI", "Individualized_Treatment_Recommendation_Rationality_ITRR"),
            ("RC", "Medical_Record_Compliance_MRC"),
            ("ED", "Physician_Patient_Communication_Caring_Effectiveness_PPCCE"),
            ("FP", "Follow_up_Prognosis_Management_FPM")
        ]
        
        for short_name, full_name in subjective_keys:
            metric = subjective_metrics.get(full_name, {})
            score = metric.get("score", 0)
            scores[short_name] = float(score / 10.0) if short_name!='CTC' else score # 转换为0-1范围
        
        return scores

    def save_results(self, results: List[Dict], output_file: str):
        """
        Save the evaluation results to a JSONL file.
        
        Args:
            results (List[Dict]): List of evaluation result dictionaries.
            output_file (str): Path to the output file.
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error: Exception occurred while saving results: {e}")

    # --- New Section: Encapsulate single case processing logic ---
    def process_single_case(self, case_data_with_index: Tuple[int, Dict]) -> Dict:
        """
        Process the complete workflow for a single medical case. 
        Acts as the unit of work for parallel tasks.
        
        Args:
            case_data_with_index (Tuple): A tuple containing (index, case_data dictionary).
            
        Returns:
            Dict: The final result dictionary including IDs, scores, and timestamp. None if failed.
        """
        index, case_data = case_data_with_index
        case_id = case_data.get('original_data', {}).get('id', f'case_{index}')
        
        try:
            # 1. Call API to get evaluation results
            evaluation_result = self.call_openai_api(case_data)
            if not evaluation_result:
                print(f"Case {case_id} API call failed, skipped.")
                return None
            
            # 2. Calculate scores for various dimensions
            scores = self.calculate_scores(evaluation_result)
            
            # 3. Encapsulate final result
            result = {
                "case_id": case_id,
                "evaluation_result": evaluation_result,
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            return result
        except Exception as e:
            print(f"Error: Exception occurred while evaluating case {case_id}: {e}")
            return None

    # Refactor evaluation loop with ThreadPoolExecutor
    def evaluate_cases(self, input_file: str, output_file: str = None):
        """
        Evaluate all cases concurrently using a thread pool.
        
        Args:
            input_file (str): Path to the input JSONL file.
            output_file (str, optional): Path to the output file. Defaults to timestamped filename.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = f"{input_path.stem}_evaluation_{timestamp}.jsonl"
        
        print(f"Starting parallel evaluation of cases...")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Using {self.max_workers} worker threads.")
        
        with jsonlines.open(input_file, 'r') as loader:
            cases = [c for c in loader]
        print(f"Successfully loaded {len(cases)} cases")
        
        results = []
        
        # Parallel processing using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a list of tuples containing index and data to submit tasks
            # enumerate(cases, 1) starts index at 1 for logging convenience
            tasks_with_indices = list(enumerate(cases, 1))

            # Submit all tasks to the thread pool
            future_to_case = {executor.submit(self.process_single_case, task): task for task in tasks_with_indices}
            
            # Process completed tasks using tqdm for progress display
            for future in tqdm.tqdm(as_completed(future_to_case), total=len(cases), desc="Evaluation Progress"):
                result = future.result() # Retrieve result of single case processing
                if result:
                    results.append(result)
                    
                    # (Optional) Periodic save logic
                    if len(results) % 10 == 0:
                        print(f"\nProcessed {len(results)} cases, saving intermediate results...")
                        self.save_results(results, output_file)
                        self.print_overall_statistics(results)

        # Final save and statistics print after all tasks are done
        if results:
            print("All cases evaluated, saving final results...")
            self.save_results(results, output_file)
            self.print_overall_statistics(results)
        else:
            print("No cases were successfully evaluated.")


    def print_overall_statistics(self, results: List[Dict]):
        """
        Print overall statistical information based on M1-Env Benchmark categories.
        
        Args:
            results (List[Dict]): List of evaluation results.
        """
        if not results:
            print("No results to evaluate.")
            return
            
        print("\n" + "="*60)
        print("M1-Env Benchmark Overall Statistics")
        print("="*60)
                
        # 1. Collect all scores
        all_scores = {}
        for result in results:
            scores = result.get("scores", {})
            for metric, score in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        # 2. Define Metric Categories based on the provided Table
        metric_categories = {
            "Clinical Efficacy": ["WC", "DR@3", "DR@5", "TP"],
            "Process Competency": ["IS", "LC", "RC", "IC", "RU"],
            "Professional Ethics": ["PS", "TI", "ED", "FP"]
        }

        # 3. Calculate and Print Average Scores per Category
        print("Average Scores (Points/Percentage):")
        
        for category_name, keys in metric_categories.items():
            print(f"\n[{category_name}]") # Print Category Header
            
            has_metrics = False
            for key in keys:
                if key in all_scores:
                    has_metrics = True
                    avg_score = sum(all_scores[key]) / len(all_scores[key])
                    
                    # logic: Check if we need to multiply by 100 based on your previous code style.
                    # Assuming raw scores are 0.0-1.0 and table shows 0-100.
                    final_score = avg_score * 100
                    
                    print(f"  {key:<6}: {final_score:.2f}")
            
            if not has_metrics:
                print(f"  (No metrics found for {category_name})")
        
        print("-" * 60)
        print(f"Total Cases Evaluated: {len(results)}")
        print("=" * 60)
        
        
def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Diagnosis Evaluation Script")
    
    parser.add_argument("--input_file", default="Path to input file", help="Path to input JSONL file")
    parser.add_argument("--output", help="Path to output file")
    
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name to use")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--api_key", default="")
    
    parser.add_argument("--max_workers", type=int, default=6, help="Number of worker threads for parallel processing")
    
    # Toggle to control whether to perform statistics only (skipping API evaluation)
    parser.add_argument("--stats_only", action="store_true", help="If set, the script will only read the output file to compute statistical metrics without performing API evaluation.")
    
    args = parser.parse_args()
    
    # Create evaluator instance with parallel worker count
    evaluator = MedicalEvaluator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_workers=args.max_workers
    )
    
    if args.stats_only:
        print(f"=== Entering Statistics Mode ===")
        print(f"Reading result file: {args.output}")
        
        results = []
        try:
            # Use jsonlines to read the completed result file
            with jsonlines.open(args.output, 'r') as loader:
                for line in loader:
                    results.append(line)
            
            print(f"Successfully loaded {len(results)} evaluation records.")
            
            # Invoke the statistical function
            if results:
                evaluator.print_overall_statistics(results)
            else:
                print("Warning: File is empty or format is incorrect.")
                
        except FileNotFoundError:
            print(f"Error: File not found: {args.output}")
        except Exception as e:
            print(f"Error: An exception occurred while reading file or calculating statistics: {e}")
            
    else:
        print(f"=== Entering Evaluation Mode ===")
        # Execute evaluation
        evaluator.evaluate_cases(args.input_file, args.output)


if __name__ == "__main__":
    main()