import os
import random
import logging
from openai import OpenAI
from pptree import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BASE_URL = None
API_KEY = None
MODEL_NAME = None
def initialize_settings(base_url: str, api_key: str, model_name: str = "gpt-4o"):
    """
    Initialize the global configuration for the program.
    This is the single entry point for all external configurations.
    """
    global BASE_URL, API_KEY, MODEL_NAME
    
    logger.info("Initializing settings...")
    BASE_URL = base_url
    API_KEY = api_key
    MODEL_NAME = model_name
    logger.info("Settings initialized successfully.")


class Agent:
    def __init__(self, instruction, examplers=None, model_name='gpt-4o', temperature=0.0):
        self.instruction = instruction
        self.model_name = MODEL_NAME
        self.temperature = temperature

        if self.model_name:
            self.client = OpenAI(
                base_url=BASE_URL, 
                api_key=API_KEY
            )
            # System prompt setup
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            # Add few-shot examples if provided
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, chat_mode=True):
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature
        )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content


class Group:
    """
    Manages a sub-group of agents for specific tasks (e.g., Initial Assessment, Final Review).
    """
    def __init__(self, goal, members, question, examplers=None):
        self.goal = 'conduct' + goal.lower().split(' team')[0]
        self.members = []
        for member_info in members:
            # Initialize agent with specific role and expertise
            _agent = Agent('You are a {} who {}. You need to review a medical case and cooperate with your group members to give an accurate and responsible diagnosis.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'])
            _agent.chat('You are a {} who {}. You need to review a medical case and cooperate with your group members to give an accurate and responsible diagnosis.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            
            # Identify leader and assistants
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            # Leader assigns tasks
            delivery_prompt = f'''You are the leader of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\n**Question**: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            # Assistants perform investigations
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group leader is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            # Leader synthesizes results
            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\n**Question**: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""
            if message:
                investigation_prompt = f"Here is the review message from previous expert groups: {message}.\n" + investigation_prompt

            response = lead_member.chat(investigation_prompt) + '\n' + lead_member.chat('Now give 2-3 sentences to support your assessment.')

            return response

        elif comm_type == 'external':
            return


def parse_group_info(group_info):
    """
    Parses the string representation of group information into a dictionary.
    """
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('@')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('@')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info


# --- MDT (Multi-Disciplinary Team) Workflow ---
def mdt(patient_case, recruited_agents) -> str:
    # recruited_agents: List. [(role, description),...]
    logger.info("MDT --- [INFO] Step 1. Expert Recruitment")

    agent_emoji = [
        '\U0001F468\U0001F3FB\u200D\u2695\uFE0F',  # 浅肤色男医生
        '\U0001F469\U0001F3FC\u200D\u2695\uFE0F',  # 中浅肤色女医生
        '\U0001F468\U0001F3FD\u200D\u2695\uFE0F',  # 中肤色男医生
        '\U0001F469\U0001F3FE\u200D\u2695\uFE0F',  # 中深肤色女医生
        '\U0001F468\U0001F3FF\u200D\u2695\uFE0F',  # 深肤色男医生
        '\U0001F469\U0001F3FF\u200D\u2695\uFE0F',  # 深肤色女医生
        '\U0001F9D1\U0001F3FB\u200D\u2695\uFE0F',  # 浅肤色中性医生
        '\U0001F9D1\U0001F3FE\u200D\u2695\uFE0F',  # 中深肤色中性医生
        ]
    random.shuffle(agent_emoji)

    # initialize each agent
    agent_list = ""
    medical_agents = []
    agents_roles = []
    for i, agent in enumerate(recruited_agents):
        agent_role = agent[0].strip().lower()
        description = agent[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
        logger.info(f"Agent {i+1} {agent_emoji[i]}: {agent_role} - {description}\n")

        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate and discuss with other medical experts in a team to make accurate final diagnosis."""
        _agent = Agent(instruction=inst_prompt)
        _agent.chat(inst_prompt)
        medical_agents.append(_agent)
        agents_roles.append(agent_role)

    logger.info("[INFO] Step 2. Collaborative Decision Making")
    logger.info("[INFO] Step 2.1. Hierarchy Selection")

    num_rounds = 2 # TODO
    num_turns = 2 # TODO
    num_agents = len(medical_agents)

    # interaction record
    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    logger.info("MDT --- [INFO] Step 2.2. Participatory Debate")

    round_opinions = {n: {} for n in range(0, num_rounds+1)} # dicts in dict

    logger.info('[INITIAL REPORT]')
    # initial opinions
    for k, agent in enumerate(medical_agents):
        opinion = agent.chat(f'''**Question**: {patient_case}\n\nAnswer and explain: ''')
        round_opinions[0][f"Agent {k+1}. "+agents_roles[k].lower()] = opinion
        logger.info(f"{k+1}. {agents_roles[k].lower()}: {opinion}")

    for n in range(1, num_rounds+1):
        logger.info(f"======= Round {n} =======")
        round_name = f"Round {n}"
    
        assessment = "".join(f"({k.lower()}): {v}\n\n" for k, v in round_opinions[n-1].items())
    
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            logger.info(f"\n|__{turn_name}")

            num_yes = 0 
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                participate_prompt = "Given the opinions from other medical experts in your team, please consider the differences between the opinions, if there are some one who share different view with you, it may be better to discuss with them. Please indicate whether you want to talk to any expert and no reasons is needed (yes/no)\n\nOpinions:\n{}".format(assessment if turn_num == 0 else all_comments)
                if turn_num != 0:
                    last_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][f"Turn {turn_num}"].items())
                    participate_prompt = '**Comments of Last Turn**:\n' + last_comments + '**Comments of This Turn**:\n' + participate_prompt
                participate = v.chat(participate_prompt)
            
                logger.info(f'\nAgent {idx+1} wants discussion? {participate}')
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please review your medical expertise and provide your opinion to the selected expert (Agent {ce}). In your discussion, highlight differences between your opinions, reflect on or revise your perspective based on other opinions if appropriate, and aim to persuade other experts. Deliver your opinion confidently, supported by a concise rationale to convince the expert.")
                    
                        logger.info(f"Agent {idx+1} ({agent_emoji[idx]}) -> Agent {ce} ({agent_emoji[ce-1]}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
            
                    num_yes += 1
                else:
                    logger.info(f" Agent {idx+1} ({agent_emoji[idx]}): \U0001f910")

            if num_yes == 0:
                break

        # tmp_final_answer = {}
        final_answer = ""
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Having engaged with other medical experts, review your expertise and their comments, then provide your final answer to the following question:\n{patient_case}\nAnswer: ")
            # tmp_final_answer[agent.role] = response
            final_answer += f'Agent {i+1}: ' + response + '\n'
            round_opinions[n][f"Agent {i+1}"] = response + '. ' + agent.chat('Now you are required to explain about why you maintain or update your diagnosis in 1-2 sentences.')
        # final_answer = tmp_final_answer
        if num_yes == 0:
            break

    logger.info("\nMDT --- [INFO] Step 3. Final Decision")

    round_opinions = [f"Round {i+1} Initial Opinions:\n" + "".join(f"({k.lower()}): {v}\n\n" for k, v in round_opinions[idx].items()) for i, idx in enumerate(round_opinions)]

    moderator = Agent(
        "You are a medical decision-making moderator tasked with reviewing opinions from multiple medical experts to deliver a final decision.", 
        )
    moderator.chat("As a medical decision-maker, your role is to review the opinions of multiple medical experts and determine the final decision.")

    _decision = moderator.chat(f"**Question**: {patient_case}\nReview each agent's final answer and determine the final decision for the question by taking a majority vote. If some options share equal votings, then make your own choice based on your exptertise instead of random choosing. **Round Initial Opinions**:{round_opinions}\n**Discussion Records**:{interaction_log}\nMake a final and clear medical diagnosis report of the case based on the discussion among the experts:\n\n")
    # final_decision = {'majority': _decision}

    health_worker_emoji = "\U0001F468\u200D\u2696\uFE0F"
    logger.info(f"{health_worker_emoji} moderator's final report: {_decision}")

    return _decision

# def organize_mdt_goups(question, recruited_agents) -> str:
#     ##### EXAMPLE
#     # recruit_prompt = f"""You are an experienced medical expert leader, you are familiar with different medical experts and their expertise. You are appointed to recruits organize Multidisciplinary Teams (MDTs) and the medical expert members in MDT and ask them to discuss and solve the given complex medical question. You are responsible and experienced to know who to recruit to make their cooperation efficient, accurate and robust."""
#     # tmp_agent = Agent(instruction=recruit_prompt, role='recruiter')
#     # tmp_agent.chat(recruit_prompt)
#     # num_teams = 3  # You can adjust this number as needed
#     # num_agents = 3  # You can adjust this number as needed
#     # recruite_prompt = f"**Case**: {question}\n\nOrganize {num_teams} multidisciplinary teams (MDTs) with distinct specialties or purposes, each consisting of {num_agents} clinicians. Based on the medical question and provided options, develop a recruitment plan to ensure an accurate and comprehensive response.\n\nFor example, a sample response could be:\nGroup 1 @ Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) @ Specializes in ear, nose, and throat surgery, including thyroidectomy, leading the team due to expertise in surgical interventions and managing complications like nerve damage.\nMember 2: General Surgeon @ Provides supplementary surgical expertise and assists in managing thyroid surgery complications.\nMember 3: Anesthesiologist @ Focuses on perioperative care, pain management, and evaluating anesthesia-related complications impacting voice or airway function.\n\nGroup 2 @ Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) @ Manages long-term care for Graves' disease, including hormonal therapy and monitoring post-surgical complications.\nMember 2: Speech-Language Pathologist @ Specializes in voice and swallowing disorders, offering rehabilitation to improve speech and voice quality post-nerve damage.\nMember 3: Neurologist @ Evaluates nerve damage and advises on recovery strategies, contributing neurological expertise.\n\nGroup 3 @ Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from Relevant Specialty (Lead) @ Offers high-level expertise and guides final decision-making.\nMember 2: Clinical Decision Specialist @ Integrates recommendations from all teams to create a cohesive treatment plan.\nMember 3: Advanced Diagnostic Specialist @ Employs advanced diagnostic tools to confirm the extent and cause of complications, supporting final decisions.\n\nThe above is an example. Create your own unique MDTs, ensuring the inclusion of the Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT). Adhere strictly to the format shown above, without explanations or comments. All groups must be fully populated, and the response must be in English only, with no other languages permitted.\n\nNow organize: "
#     # recruited = tmp_agent.chat(recruite_prompt)

#     groups = [group.strip() for group in recruited.split("Group") if group.strip()]
#     group_strings = ["Group " + group for group in groups]

#     group_instances = []
#     for i1, gs in enumerate(group_strings):
#         res_gs = parse_group_info(gs)
#         group_instance = Group(res_gs['group_goal'], res_gs['members'], question, fewshot_examplers)
#         group_instances.append(group_instance)

#     logger.info('[STEP 2] Initial Assessment from Each Group')
#     # STEP 2. initial assessment from each group
#     # STEP 2.1. IAP Process
#     logger.info('[STEP 2.1] IAP Process')
#     initial_assessments = []
#     for group_instance in group_instances:
#         if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
#             init_assessment = group_instance.interact(comm_type='internal', )
#             initial_assessments.append([group_instance.goal, init_assessment])

#     initial_assessment_report = ""
#     if len(initial_assessments)==0:
#         initial_assessment_report = "None"
#     for idx, init_assess in enumerate(initial_assessments):
#         initial_assessment_report += f"Initial Assessment {idx+1}: {init_assess[0]} - {init_assess[1]}. \n"
#     logger.info(initial_assessment_report)

#     # STEP 2.2. other MDTs Process
#     logger.info('[STEP 2.2] Other MDTs Process')
#     assessments = []
#     for group_instance in group_instances:
#         if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
#             assessment = group_instance.interact(comm_type='internal',message=initial_assessment_report)
#             assessments.append([group_instance.goal, assessment])

#     assessment_report = ""
#     for idx, assess in enumerate(assessments):
#         assessment_report += f"MDTs Group {idx+1}: {assess[0]} - {assess[1]}\n"
#     logger.info(assessment_report)

#     # STEP 2.3. FRDT Process
#     logger.info('[STEP 2.3] Final Review and Deciding Process')
#     final_decisions = []
#     for group_instance in group_instances:
#         if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
#             decision = group_instance.interact(comm_type='internal',message=initial_assessment_report + assessment_report)
#             final_decisions.append([group_instance.goal, decision])

#     compiled_report = ""
#     for idx, decision in enumerate(final_decisions):
#         compiled_report += f"Final Review and Decision {idx+1}: {decision[0]} - {decision[1]}\n"
#     logger.info(compiled_report)

#     # STEP 3. Final Decision
#     logger.info('[STEP 3] Final Decision')
#     decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
#     tmp_agent = Agent(instruction=decision_prompt, role='decision maker')
#     tmp_agent.chat(decision_prompt)

#     final_decision = tmp_agent.temp_responses(f"""**Case**: {question}\n**Initial Assessment**:\n{initial_assessment_report}\n**Further Medical Assesments**:{assessment_report}\n**Final Assessment**:{compiled_report}.\nNow please make your final and clear diagnosis report on the case: """, img_path=None)
#     logger.info('[Final Decision]', final_decision)
#     return final_decision

if __name__ == "__main__":

    # Patient case example
    patient_case = """{
        'clinic_note': 'Patient presents with progressive necrotic scrotal ulcers, fever, spontaneous bruising, epistaxis, 
hematuria, and recent low white blood cell count. Associated symptoms include scrotal swelling, warmth, and erythema, along with
non-pruritic skin spots. Concerns include possible vasculitis, infectious etiology (e.g., septic emboli, anthrax, or other 
systemic infection), or a hematologic disorder with secondary infection. The combination of necrotic skin lesions, systemic 
signs of infection, and hematologic abnormalities is highly concerning.',
        'suggestion': 'Infectious Disease',
        'next_step': 'specialist_consultation',
        'Recording Department': 'registration'
    },
    {
        'clinic_note': 'The patient presents with a rapidly progressive clinical syndrome characterized by necrotic scrotal 
ulcers with eschar formation, fever, spontaneous bleeding (bruising, epistaxis, hematuria), transient non-pruritic skin lesions 
(possibly livedo reticularis), and a history of profound neutropenia with partial recovery. Imaging confirms deep tissue 
necrosis in the scrotum without evidence of torsion. The absence of trauma, IV drug use, or recent invasive procedures makes 
nosocomial or injection-related causes less likely. The combination of systemic signs of infection, necrotizing soft tissue 
involvement, and hematologic abnormalities—particularly in the context of neutropenia—strongly suggests a severe, toxin-mediated
bacterial infection. The clinical picture is most consistent with necrotizing fasciitis due to *Clostridium perfringens* or 
another gas-forming, rapidly progressive anaerobic organism, especially given the necrotic ulcers, systemic toxicity, and 
progression despite neutrophil recovery. The absence of typical risk factors for anthrax (e.g., animal exposure) and the 
presence of systemic vasculitic-like skin changes (livedo reticularis) further support an infectious etiology with 
toxin-mediated tissue destruction. This is not a primary vasculitis or hematologic disorder, but rather a secondary infection 
superimposed on neutropenia.',
        'suggestion': 'Immediate empiric treatment with broad-spectrum antibiotics covering anaerobes and Gram-positive 
organisms, including penicillin G, clindamycin, and vancomycin, is indicated. Surgical debridement of necrotic tissue in the 
scrotum is critical. Blood cultures, wound cultures (including anaerobic), and tissue biopsy for histopathology and microbiology
(including PCR for *Clostridium* spp.) must be obtained immediately. A CT scan of the abdomen and pelvis should be performed to 
assess for deep-seated infection or gas formation. If the clinical course deteriorates or imaging reveals gas, consider 
hyperbaric oxygen therapy. The patient requires ICU-level monitoring and multidisciplinary management.',
        'next_step': 'diagnostic_test',
        'Recording Department': 'specialist_consultation'
    },
    {'Recording Department': 'diagnostic_test', 'Completed Tests': "['Blood cultures']"},
    {
        'clinic_note': 'The patient is a previously healthy individual who developed profound neutropenia (neutrophil count 0.3 
× 10⁹/L) following spontaneous bruising and epistaxis, prompting hospitalization. Within days of neutropenia onset, rapidly 
progressive necrotic scrotal ulcers with eschar formation developed, accompanied by fever, hematuria, and persistent bleeding. 
The clinical course is characterized by deep tissue necrosis, absence of trauma or risk factors for anthrax, and no evidence of 
vasculitis or primary hematologic disorder. The combination of severe neutropenia, rapidly progressive necrotizing soft tissue 
infection with eschar, systemic toxicity, and lack of response to immune recovery strongly suggests a toxin-mediated, 
gas-forming anaerobic infection. The absence of crepitus and gastrointestinal symptoms does not rule out *Clostridium 
perfringens*, which can cause necrotizing fasciitis without gas in some cases, particularly in immunocompromised hosts. The 
clinical picture is most consistent with necrotizing fasciitis due to *Clostridium perfringens*, a condition that can present 
with rapid tissue destruction, systemic toxicity, and high mortality if not treated immediately. Blood cultures have been 
obtained, and further diagnostic testing is ongoing.',
        'suggestion': 'Immediate surgical debridement of necrotic scrotal tissue is critical. Empiric antibiotic therapy should 
be continued and optimized with high-dose penicillin G (12–24 million units/day IV), clindamycin (900 mg IV every 8 hours), and 
vancomycin (15 mg/kg IV every 12 hours) to cover *Clostridium perfringens* and other Gram-positive anaerobes. A CT scan of the 
abdomen and pelvis should be performed urgently to assess for gas formation, deep-seated abscesses, or extension into the 
retroperitoneum. Tissue biopsy for histopathology, Gram stain, and anaerobic culture (including PCR for *Clostridium* spp.) is 
essential. If gas is detected on imaging or clinical deterioration occurs, hyperbaric oxygen therapy should be considered. The 
patient requires ICU-level monitoring due to high risk of sepsis and multiorgan failure.',
        'next_step': 'diagnostic_test',
        'Recording Department': 'specialist_consultation'
    }
]
[
    {
        'test_name': 'Blood cultures',
        'test_result': 'Blood cultures: Negative for bacterial, fungal, and mycobacterial growth after 7 days of incubation.'
    }
]"""

    recruited_agents = [
        ("Medical Oncologist", 
        "Responsible for developing systemic treatment plans, including chemotherapy, targeted therapy, immunotherapy, and endocrine therapy, and assessing the patient's overall condition."),
        
        ("Surgical Oncologist", 
        "Assesses whether tumors can be surgically resected and develops specific surgical plans; central to local treatment."),
        
        ("Radiologist", 
        "Clarifies tumor location, size, relationship with surrounding tissues, and existence of metastases by interpreting medical images like CT, MRI, and PET-CT."),
        
        ("Pathologist", 
        "Provides the final diagnosis ('Gold Standard') of the tumor, including type, grading, and molecular characteristics, by examining biopsy or surgically resected tissue samples under a microscope."),
    ]
    
    initialize_settings(
        base_url="YOUR_URL",
        api_key="YOUR_KEY",
        model_name="YOUR_MODEL"
    )
    
    result = mdt(patient_case, recruited_agents)
    
    logger.info(result)
