"""
Medical Diagnosis Simulation Pipeline

This module orchestrates a multi-agent simulation of a medical diagnosis process.
It manages the interaction between a 'Patient Agent' (simulating a specific medical case based on real data) and various 'Doctor Agents' (Registration, Specialist, etc.) to traverse a hospital workflow from initial presentation to final diagnosis.
"""


import re
import json
import logging
import traceback

from openai import OpenAI
from tqdm import tqdm
from rich import print
from concurrent.futures import ThreadPoolExecutor

from utils.tool_manager import ToolManager
from utils.tools import Tools
from utils.agents import Prompts, PatientAgent, DoctorAgent


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inference import Args
    from utils.run_context import RunContext
    

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def talk_with_doctor(args: "Args", context: "RunContext", doctor_role: str, encounter_turns: int) -> bool:
    """
    Manages a single interaction encounter between the Patient Agent and a Doctor Agent.
    
    Args:
        args: Configuration arguments.
        context: Runtime context containing history contents.
        doctor_role: The specific role of the doctor.
        encounter_turns: The current count of encounters in the pipeline.
        
    Returns:
        bool: True if the interaction finished successfully (reached a conclusion or completed tests), False if max turns exceeded.
    """
    
    # --- Initialize Doctor Agent ---
    assert doctor_role in Prompts.roles
    
    # Configure the Doctor Agent based on the role.
    # Specialists get access to external tools (if configured), while others might not.
    if doctor_role == "specialist_consultation":
        if not args.doctor_responses_api:
            doctor_agent = DoctorAgent(
                llm_client=context.doctor_client,
                llm_model=args.doctor_model_name,
                tools_json=context.tool_manager.get_tools_json(),
                tool_manager=context.tool_manager,
                responses_api=args.doctor_responses_api
            )
        else:
            doctor_agent = DoctorAgent(
                llm_client=context.doctor_client,
                llm_model=args.doctor_model_name,
                tools_json=context.tool_manager.get_tools_json_responses_api(),
                tool_manager=context.tool_manager,
                responses_api=args.doctor_responses_api
            )
        # print(tool_manager.get_tools_json())
    else:
        # General roles usually don't need complex tools
        doctor_agent = DoctorAgent(
            llm_client=context.doctor_client,
            llm_model=args.doctor_model_name,
            responses_api=args.doctor_responses_api
        )
    
    # --- Branch: Diagnostic Testing Phase ---
    # If the current role is 'diagnostic_test', we simulate running medical tests instead of a conversation.
    if doctor_role == "diagnostic_test":
        task = str(context.patient_records[-1]["suggestion"])
        
        # 1. Identify required tests.
        test_itemize_agent = PatientAgent(llm_client=context.patient_agent.llm_client, llm_model=args.patient_model_name)
        test_itemize_agent.history_messages = []
        tests = test_itemize_agent.chat(Prompts.prompts[doctor_role]["itemize"].replace("INSERT_TESTS_HERE", task))
        tests = re.findall(r"```json(.*?)```", tests, flags=re.DOTALL)[0].strip()
        tests = json.loads(tests)
        
        # 2. Define helper to simulate a single test execution.
        def mimic_test(test_name: str):
            test_agent = PatientAgent(llm_client=context.patient_client, llm_model=args.patient_model_name)
            test_instruction = Prompts.prompts[doctor_role]["test"].replace("INSERT_TEST_HERE", test_name).replace("INSERT_CASE_HERE", context.patient_agent.whole_case)
            result = test_agent.chat(test_instruction)
            
            # # (Optional) Generate a formal report from the raw result by the Doctor.
            # report_agent = DoctorAgent(llm_client=context.doctor_client, llm_model=args.doctor_model_name)
            # report_instruction = Prompts.prompts[doctor_role]["report"].replace("INSERT_REPORT_HERE", result)
            # report = report_agent.chat(report_instruction)
            
            context.diagnosis_reports.append({
                "test_name": test_name,
                "test_result": result,
                # "test_report": report
            })
            logger.info(f"\n--- {doctor_role} ---:  {context.diagnosis_reports[-1]}\n")
            
        # 3. Execute tests concurrently to simulate parallel lab work.
        with ThreadPoolExecutor(max_workers = 8) as executor:
            list(executor.map(mimic_test, tests))
        
        # 4. Record the completed tests in the patient records.
        context.patient_records.append({
            "Recording Department": "diagnostic_test",
            "Completed Tests": str(tests)
        })
        context.patient_records_simplified.append({
            "Recording Department": "diagnostic_test",
            "Completed Tests": str(tests)
        })
        
        # 5. Log the interaction as a encounter turn.
        context.doctor_patient_diaglogues.append({
            "encounter_turn": encounter_turns,
            "department": doctor_role,
            "diaglogue": str(tests)
        })
        return True
        
    # --- Branch: Interactive Dialogue Phase ---
    # Main loop for conversation between Doctor and Patient.
    now_diaglogue = []
    for i in range(args.max_talk_turns+3):
        
        # Determine if the doctor is allowed to call tools (only specialists).
        tool_call = False
        if doctor_role == "specialist_consultation":
            tool_call = True
        
        # Initialize storage for the current turn's dialogue data.
        now_talkturn_diaglogue = {}
        # logger.info(f"\n--- now_diaglogue --- \n{now_diaglogue}\n")
        # logger.info(f"\n--- {doctor_role} --- turns {i+1} start!\n")
            
        # --- First Turn & Initial Persona Setup ---
        if i == 0:
            if doctor_role == "registration":
                instruction = Prompts.prompts[doctor_role]["doctor"].replace("INSERT_QUERYNUMS_HERE", str(args.max_query_nums)).replace("INSERT_TURNS_HERE",str(args.max_talk_turns))
            elif doctor_role == "specialist_consultation":
                instruction = Prompts.prompts[doctor_role]["doctor"].replace("INSERT_QUERYNUMS_HERE", str(args.max_query_nums)).replace("INSERT_TURNS_HERE",str(args.max_talk_turns))
                instruction = instruction.replace("INSERT_DEPARTMENT_HERE", context.patient_records[1]["suggestion"]).replace("INSERT_RECORD_HERE", str(context.patient_records_simplified))
            logger.info(f"\n--- {doctor_role} --- doctor_prompt ---:  {instruction}\n")
            
            # Generate the Doctor's opening statement.
            doctor_talk = doctor_agent.chat(instruction)
            now_talkturn_diaglogue["turn"] = i + 1
            now_talkturn_diaglogue["doctor"] = doctor_talk
            logger.info(f"\n---{doctor_role} --- doctor_talk---:  {doctor_talk}\n")
            
            # Clean up internal 'Thought' processes (<think> tags) before showing text to the Patient Agent.
            if "<think>" in doctor_talk and "</think>" in doctor_talk:
                doctor_talk_to_patient = re.sub('<think>.*</think>', '', doctor_talk, flags=re.DOTALL).strip()
            elif "<seed:think>" in doctor_talk and "</seed:think>" in doctor_talk:
                doctor_talk_to_patient = re.sub('<seed:think>.*</seed:think>', '', doctor_talk, flags=re.DOTALL).strip()
            else:
                doctor_talk_to_patient = doctor_talk

            # Check if the doctor has reached a conclusion immediately (Rare but possible in Turn 1).
            if re.search(r"```json(.*?)```", doctor_talk_to_patient, flags=re.DOTALL):
                report = re.findall(r"```json(.*?)```", doctor_talk_to_patient, flags=re.DOTALL)[0].strip()
                report = json.loads(report)

                report.update({"Recording Department": doctor_role})
                context.patient_records.append(report)
                
                report_simplified = {
                    "Recording Department": doctor_role,
                    "Clinic Note": report["clinic_note"]
                }
                context.patient_records_simplified.append(report_simplified)
                
                now_diaglogue.append(now_talkturn_diaglogue)
                context.doctor_patient_diaglogues.append({
                    "encounter_turn": encounter_turns,
                    "department": doctor_role,
                    "diaglogue": now_diaglogue
                })
                
                # If diagnosis is complete, generate the final discharge summary.
                if report["next_step"] == "end_of_diagnosis":
                    result = doctor_agent.chat(Prompts.prompts["end_of_diagnosis"])
                    result = re.findall(r"```json(.*?)```", result, flags=re.DOTALL)[0].strip()
                    result = json.loads(result)
                    context.patient_records.append({
                        "Recording Department": "end_of_diagnosis",
                        "final_diagnosis": result["final_diagnosis"],
                        "treatment_plan": result["treatment_plan"]
                    })
                return True
            
            # Generate Patient's response to the opening statement.
            if doctor_role == "registration":
                instruction = Prompts.prompts[doctor_role]["patient"].replace("INSERT_QUERY_HERE", doctor_talk_to_patient)
                patient_talk = context.patient_agent.chat(instruction)
                # logger.info(f"\n--- {doctor_role} --- patient_prompt ---:  {instruction}\n")
            elif doctor_role == "specialist_consultation":
                instruction = Prompts.prompts[doctor_role]["patient"].replace("INSERT_QUERY_HERE", doctor_talk_to_patient).replace("INSERT_DEPARTMENT_HERE", context.patient_records[1]["suggestion"])
                patient_talk = context.patient_agent.chat(instruction)
                # logger.info(f"\n--- {doctor_role} --- patient_prompt ---:  {instruction}\n")
            now_talkturn_diaglogue["patient"] = patient_talk
            logger.info(f"\n--- patient_talk ---:  {patient_talk}\n")
            
        # --- Subsequent Dialogue Turns ---
        else:
            now_tool_usage_info = []
            doctor_talk = doctor_agent.chat(patient_talk, now_tool_usage_info, tool_call)

            now_talkturn_diaglogue["turn"] = i + 1
            
            # Record tool usage if any occurred.
            if now_tool_usage_info:
                now_talkturn_diaglogue["tool"] = now_tool_usage_info.copy()
                for tool_info in now_tool_usage_info:
                    tool_info["encounter_turn"] = encounter_turns
                    context.tool_usages.append(tool_info)
            now_talkturn_diaglogue["doctor"] = doctor_talk
            logger.info(f"\n--- {doctor_role} --- doctor_talk ---:  {doctor_talk}\n")

            # Clean up internal 'Thought' processes again.
            if "<think>" in doctor_talk and "</think>" in doctor_talk:
                doctor_talk_to_patient = re.sub('<think>.*</think>', '', doctor_talk, flags=re.DOTALL).strip()
            elif "<seed:think>" in doctor_talk and "</seed:think>" in doctor_talk:
                doctor_talk_to_patient = re.sub('<seed:think>.*</seed:think>', '', doctor_talk, flags=re.DOTALL).strip()
            else:
                doctor_talk_to_patient = doctor_talk
    
            # Check if Doctor has output a JSON conclusion.
            if re.search(r"```json(.*?)```", doctor_talk_to_patient, flags=re.DOTALL):
                report = re.findall(r"```json(.*?)```", doctor_talk_to_patient, flags=re.DOTALL)[0].strip()
                report = json.loads(report)

                report.update({"Recording Department": doctor_role})
                context.patient_records.append(report)
                
                report_simplified = {
                    "Recording Department": doctor_role,
                    "Clinic Note": report["clinic_note"]
                }
                context.patient_records_simplified.append(report_simplified)
                
                now_diaglogue.append(now_talkturn_diaglogue)
                context.doctor_patient_diaglogues.append({
                    "encounter_turn": encounter_turns,
                    "department": doctor_role,
                    "diaglogue": now_diaglogue
                })
                
                # If diagnosis is complete, generate the final discharge summary.
                if report["next_step"] == "end_of_diagnosis":
                    result = doctor_agent.chat(Prompts.prompts["end_of_diagnosis"])
                    result = re.findall(r"```json(.*?)```", result, flags=re.DOTALL)[0].strip()
                    result = json.loads(result)
                    context.patient_records.append({
                        "Recording Department": "end_of_diagnosis",
                        "final_diagnosis": result["final_diagnosis"],
                        "treatment_plan": result["treatment_plan"]
                    })
                return True
            
            # If no conclusion, the Patient responds to continue the dialogue.
            patient_talk = context.patient_agent.chat(doctor_talk_to_patient)
            now_talkturn_diaglogue["patient"] = patient_talk
            logger.info(f"\n--- patient_talk ---:  {patient_talk}\n")
            
        now_diaglogue.append(now_talkturn_diaglogue)
        
    logger.info("\n------- Out of current diaglogues! --------\n")
    return False
    # raise ValueError("Exceeded dialogue turn limit.")
  

def run_pipeline(args: "Args", context: "RunContext") -> "RunContext":
    """
    Orchestrates the entire simulation pipeline for a specific medical case.
    Initializes agents, tools, and manages the lifecycle of encounters.
    """
    
    # --- Register Patient Agent ---
    # Construct case information string for the agent's context.
    known_case = f"Case Information: {context.data_item['Case Information']}\n\nPhysical Examination: {context.data_item['Physical Examination']}"
    whole_case = f"Case Information: {context.data_item['Case Information']}\n\nPhysical Examination: {context.data_item['Physical Examination']}\n\nDiagnostic Tests: {context.data_item['Diagnostic Tests']}\n\nFinal Diagnosis: {context.data_item['Final Diagnosis']}"
    
    patient_client = OpenAI(
        base_url=args.patient_base_url,
        api_key=args.patient_api_key
    )
    patient_agent = PatientAgent(
        llm_client=patient_client,
        llm_model=args.patient_model_name,
        known_case=known_case,
        whole_case=whole_case
    )
    context.patient_client = patient_client
    context.patient_agent = patient_agent
    
    # --- Register Doctor Client ---
    doctor_client = OpenAI(
        base_url=args.doctor_base_url,
        api_key=args.doctor_api_key
    )
    context.doctor_client = doctor_client
    
    # --- Initialize Medical Record Systems ---
    # Initialize lists to track the progression of the case.
    context.patient_records = [{
        "Recording Department": "diagnostic_test",
        "Completed Tests": "[]"
    }]
    context.patient_records_simplified = [{
        "Recording Department": "diagnostic_test",
        "Completed Tests": "[]"
    }]
    context.diagnosis_reports = []
    context.tool_usages = []
    context.doctor_patient_diaglogues = []
    
    # --- Register Tools ---
    tool_manager = ToolManager()
    tools_instance = Tools(args, context)
    
    # Bind specific tools to the manager (e.g., PubMed, Wikipedia).
    tool_manager.register_tool(tools_instance.get_info)
    tool_manager.register_tool(tools_instance.search_wikipedia)
    tool_manager.register_tool(tools_instance.search_pubmed)
    tool_manager.register_tool(tools_instance.organize_mdt)
    context.tool_manager = tool_manager
    # logger.info(f"\n --- tool schema --- \n{tool_manager.get_tools_json()}\n")
    # logger.info(f"\n --- tool schema --- \n{tool_manager.get_tools_json_responses_api()}\n")
    
    # --- Simulate Medical Encounter Process ---
    encounter_turns = 1
    now_doctor_role = "registration" # Starting point: Registration desk
    
    while encounter_turns <= args.max_encounter_nums:
        # Attempt to run the encounter logic.
        try:
            s = talk_with_doctor(args, context, now_doctor_role, encounter_turns)
        except Exception as e:
            print(f"\n--- ERROR DETECTED in worker thread of data {context.data_item['id']}---")
            traceback.print_exc()
            print("---------------------------------------------------\n")
            
            context.process_success = False
            return context

        # If the conversation exceeded max turns without conclusion, fail the process.
        if not s:
            context.process_success = False
            return context
        
        # Check the result of the encounter to determine the next step.
        record = context.patient_records[-1]
        if record["Recording Department"] == "end_of_diagnosis":
            context.process_success = True
            return context
        else:
            # Determine next department: If currently testing, go to Specialist, otherwise follow the doctor's referral.
            now_doctor_role = "specialist_consultation" if now_doctor_role == "diagnostic_test" else record["next_step"]
        
        logger.info(f"\n--- {encounter_turns}th dialogue---")
        logger.info(f"\n--- next department --- {now_doctor_role}\n")
        logger.info(f"\n--- patient_records ---:  {context.patient_records}\n")
        logger.info(f"\n--- patient_records_simplified ---:  {context.patient_records_simplified}\n")
        logger.info(f"\n--- diagnosis_reports ---:  {context.diagnosis_reports}\n")
        
        encounter_turns += 1
        
    # If the loop finishes without reaching "end_of_diagnosis", mark as failure.
    context.process_success = False
    return context
        
    