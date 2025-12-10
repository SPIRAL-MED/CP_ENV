import json
import logging
import openai
from rich import print
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Prompts:

    prompts = {}
    roles = [
        "registration", "diagnostic_test", "specialist_consultation", "end_of_diagnosis"
    ]

    prompts["registration"] = {}
    prompts["registration"]["patient"] = \
"""
You have just arrived at the hospital. Your first step is to provide basic personal information to register. A guidance nurse will speak with you to get a general understanding of your condition and recommend an appropriate department.

Now the guidance nurse says: {INSERT_QUERY_HERE}
"""
    prompts["registration"]["doctor"] = \
"""
You are a hospital guidance assistant stationed in the main lobby. Your job is to briefly assess each patient's general symptoms and recommend the appropriate department for consultation. Remember, you are not a doctor — just a guide — so keep the inquiry simple and focused on directing the patient efficiently.

You may ask up to {INSERT_QUERYNUMS_HERE} questions per turn, with a maximum of {INSERT_TURNS_HERE} rounds of dialogue. You must provide your final recommendation before the dialogue ends.

Once you have reached a conclusion, respond in the following JSON format, enclosed by ```json and ```:
```json
{
    "clinic_note": "A guide note of the patient's reported symptoms and the reason for the referral.",
    "suggestion": "The single, most appropriate department for the patient to visit. Must be one department name only.",
    "next_step": "specialist_consultation"
}
```
"""

    prompts["diagnostic_test"] = {}
    prompts["diagnostic_test"]["itemize"] = \
"""
Here is the patient's diagnostic test names: {INSERT_TESTS_HERE}. Please split it into individual test names.

Output in JSON format:
```json
["Test Name 1", "Test Name 2", ...]
```
"""
    prompts["diagnostic_test"]["test"] = \
"""
You are a physician specializing in diagnostic tests and imaging. The patient has been referred to you for the following test result: {INSERT_TEST_HERE}.

Your task is to provide a complete and standard result for this test. Based on the patient's full case history below, use the existing results if they are provided. If not, generate plausible and medically appropriate results consistent with the patient's condition.

Output the test result only. Do not include any analysis or explanation.

Patient's Case:
{INSERT_CASE_HERE}
"""
    prompts["diagnostic_test"]["report"] = \
"""
You are a physician specializing in the interpretation of diagnostic tests and imaging. Your task is to provide a brief, summary comment on the results from the patient's report.

Here is the patient's report: {INSERT_REPORT_HERE}
"""

    prompts["specialist_consultation"] = {}
    prompts["specialist_consultation"]["patient"] = \
"""
You have now arrived at the specialist consultation department of {INSERT_DEPARTMENT_HERE}. 

Now the physician asks: {INSERT_QUERY_HERE}

Begin role-playing as the patient!
"""
    prompts["specialist_consultation"]["doctor"] = \
"""
You are a specialist physician in the {INSERT_DEPARTMENT_HERE} department, responsible for conducting hospital consultations. Your task is to evaluate the patient's condition through dialogue and ultimately provide a diagnosis or recommend the necessary diagnostic tests.

Patient's Medical Record: {INSERT_RECORD_HERE}.
You have the tool to get completed test reports mentioned in the medical record. If a test you need is not available, you should list the required tests in your final response.

You may ask up to {INSERT_QUERYNUMS_HERE} questions per turn, with a maximum of {INSERT_TURNS_HERE} rounds of dialogue. You must deliver your final diagnosis before the dialogue ends.

Once you have reached a conclusion, respond in the following JSON format, enclosed by ```json and ```:
```json
{
    "clinic_note": "A comprehensive clinic note for the patient's current visit. This should include your clinical assessment, the final diagnosis, and the proposed management or treatment plan.",
    "suggestion": "Your professional recommendation. If further tests are required, list them and set `next_step` to `diagnostic_test`. If the final diagnosis is confirmed, outline the diagnosis and treatment plan, and set `next_step` to `end_of_diagnosis`.",
    "next_step": "Specify one of the following options: 'diagnostic_test' or 'end_of_diagnosis'."
}
```
"""

    prompts["end_of_diagnosis"] = \
"""
Based on your final analysis, enumerate the top 5 most likely diagnoses for this patient, ordered from most to least probable. In addition, provide the definitive treatment plan.

Output in JSON format, enclosed by ```json and ```:
```json
{
    "final_diagnosis": ["Disease 1", "Disease 2", "Disease 3", "Disease 4", "Disease 5"],
    "treatment_plan": "A treatment plan for the patient"
}
```
"""



class PatientAgent:
    def __init__(self, llm_client, llm_model: str="gpt-4o", known_case: str="", whole_case: str=None) -> None:
        self.llm_model = llm_model
        self.llm_client = llm_client
        self.known_case = known_case
        self.whole_case = whole_case

        self.system_prompt = \
"""
You are a simulated patient, intended to test the hospital's medical procedures and the doctor's diagnostic skills. You are currently role-playing as a patient at a hospital, where you will interact with various individuals and engage in limited communication with them.

Below is the simulated case provided to you: {INSERT_CASE_HERE}

Please remember the following:
1. When the doctor inquires about your medical condition, you should respond based on the provided simulated case.
2. You only need to answer the questions the doctor asks you. If a question is not asked, you do not need to provide any information.
"""
        self.system_prompt = self.system_prompt.replace("INSERT_CASE_HERE", self.known_case)

        self.history_messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, message: str) -> str:
        self.history_messages.append({"role": "user", "content": message})
        response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=self.history_messages
        )

        self.history_messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        return response.choices[0].message.content
    
    def add_history_message(self, role, content, thinking=None):
        self.history_messages.append({"role": role, "content": content})
        if thinking:
            self.history_messages_with_reasoning.append({"role": role, "content": f"<think>\n{thinking}\n</think>{content}"})

    

class DoctorAgent:
    def __init__(self, llm_client, llm_model: str="gpt-4o", tools_json: str=None, tool_manager=None, responses_api=False) -> None:
        self.llm_model = llm_model
        self.llm_client = llm_client
        self.tools = json.loads(tools_json) if tools_json else None
        self.tool_manager = tool_manager

        # Boolean flag to determine if we use the standard ChatCompletion API or a custom responses API
        self.responses_api = responses_api

        # assert doctor_role in Prompts.roles
        # self.doctor_role = doctor_role
        
        # --- System Prompt Configuration ---
        self.system_prompt = \
"""
You are a doctor working in a hospital, and you are about to see a patient. You must communicate with them to inquire about their condition.
"""

        self.history_messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        self.history_messages_with_reasoning = [
            {"role": "system", "content": self.system_prompt}
        ]


    @retry(
        # Retry configuration:
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(8),
        retry=retry_if_exception_type((
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.BadRequestError
        )),
        before_sleep=lambda retry_state: logging.info(f"Retrying API call due to {retry_state.outcome.exception()}, attempt #{retry_state.attempt_number}...")
    )
    def _call_llm_with_retry(self, **kwargs):
        """
        Wrapper method for API calls to apply the tenacity retry logic defined above.
        """
        if not self.responses_api:
            # Standard OpenAI format
            return self.llm_client.chat.completions.create(**kwargs)
        else:
            # Custom responses API format
            # print(" -------------------------------- _call_llm_with_retry  -------------------------")
            # print(kwargs)
            if 'messages' in kwargs:
                kwargs['input'] = kwargs.pop('messages')
            return self.llm_client.responses.create(**kwargs)
        
    def chat(self, message: str, tool_usage_info: list=None, tool_call: bool=False) -> str:
        """
        Main chat interface for the Doctor Agent.
        
        Args:
            message (str): The user/patient input message.
            tool_usage_info (list): A list to store details about tool execution for logging/debugging.
            tool_call (bool): Whether tools are enabled for this turn.
        """
        # Maintain conversation history
        self.history_messages.append({"role": "user", "content": message})
        self.history_messages_with_reasoning.append({"role": "user", "content": message})
        
        tool_calling = False if self.tools is None else tool_call
        
        # --- Scenario A: Tools are NOT enabled for this turn ---
        if not tool_calling:
            response = self._call_llm_with_retry(
                model=self.llm_model,
                messages=self.history_messages
            )
            if not self.responses_api:
                output_text = response.choices[0].message.content
                self.history_messages.append({"role": "assistant", "content": output_text})
            else:
                output_text = response.output_text
                self.history_messages.append({"role": "assistant", "content": output_text})
                # logger.info(f"\n --- error history_messages --- \n")
                # logger.info(self.history_messages)
            return output_text
        
        # --- Scenario B: Tools ARE enabled for this turn ---
        else:
            # Logic for standard `chat.completions` API
            if not self.responses_api:
                max_tool_turns = 1
                
                turns = 0
                while True:
                    # If within allowed tool turns, pass `tools` definition to LLM
                    if turns < max_tool_turns:
                        response = self._call_llm_with_retry(
                            model=self.llm_model,
                            messages=self.history_messages,
                            tools=self.tools,
                        )
                    elif turns >= max_tool_turns:
                        response = self._call_llm_with_retry(
                            model=self.llm_model,
                            messages=self.history_messages,
                        )
                    else:
                        raise ValueError(f"turns value {turns} is error.")
                    
                    ## --- Case: Model decides NOT to call a tool ---
                    if not response.choices[0].message.tool_calls:
                        self.history_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                        # Return final text response and exit loop
                        return response.choices[0].message.content
                    
                    ## --- Case: Model decides to CALL a tool ---
                    elif response.choices[0].message.tool_calls:
                        # print("tool using")
                        turns += 1
                        
                        # Add the assistant's tool call request to history
                        self.history_messages.append({"role": "assistant", "tool_calls": response.choices[0].message.tool_calls})
                        
                        for tool_call in response.choices[0].message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            
                            # Execute the tool via tool_manager
                            tool_call_result = self.tool_manager.call_tool(tool_name, tool_args)
                            
                            # Append the tool result to history
                            self.history_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": tool_call_result
                            })
                            
                            # Log usage info
                            tool_usage_info.append({
                                "name": tool_name,
                                "id": tool_call.id,
                                "args": tool_args,
                                "content": tool_call_result
                            })
                            
                            logger.info(f"\n--- function calling ---:  {tool_name} \n {tool_call_result}\n")
                            # print(f"\n--- function calling ---:  {tool_name} \n {tool_call_result}\n")
                    else:
                        print(response.choices[0].message.tool_calls)
                        raise KeyError("Doctor Agent abnomarl chat.")  
                    
            # Logic for custom `responses` API
            else:
                max_tool_turns = 1
                
                turns = 0
                while True:
                    if turns < max_tool_turns:
                        # logger.info(f"\n--- tools ---: {self.tools}\n")
                        response = self._call_llm_with_retry(
                            model=self.llm_model,
                            messages=self.history_messages,
                            tools=self.tools
                        )
                        # logger.info(f"\n--- response.output ---: {response.output}\n")
                        # logger.info(f"\n--- response.output ---: {response.output_text}\n")
                    elif turns >= max_tool_turns:
                        response = self._call_llm_with_retry(
                            model=self.llm_model,
                            messages=self.history_messages
                        )
                    else:
                        raise ValueError(f"turns value {turns} is error.")
                    
                    # self.history_messages += response.output
                    
                    # Iterate through response items (messages, reasoning, function calls)
                    for item in response.output:
                        ## --- Case: Text Message ---
                        if item.type == "message":
                            # Final response received, return text
                            # self.history_messages.append(item)
                            self.history_messages.append({"role": "assistant", "content": response.output_text})
                            return response.output_text
                        
                        ## --- Case: Reasoning/Thinking ---
                        elif item.type == "reasoning":
                            # 'reasoning' items cannot always be added directly to history_messages depending on API strictness
                            if item.content is None:
                                continue
                            else:
                                self.history_messages.append(item)
                                
                        ## --- Case: Function Call ---
                        elif item.type == "function_call":
                            turns += 1
                            # logger.info(f"\n --- error item.name --- \n{item.name}\n")
                            # logger.info(f"\n --- error item.arguments --- \n{type(item.arguments)}\n")
                            # logger.info(f"\n --- error item.arguments --- \n{json.loads(item.arguments)}\n")
                            
                            # Execute tool
                            result_name, tool_call_result = self.tool_manager.call_tool(item.name, json.loads(item.arguments))
                            # logger.info(f"\n --- error result_name --- \n{result_name}\n")
                            # logger.info(f"\n --- error tool_call_result --- \n{tool_call_result}\n")
                            
                            # Add function call and result to history
                            self.history_messages.append(item)
                            self.history_messages.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps({result_name: tool_call_result})
                            })
                            # logger.info(f"\n --- error history_messages --- \n")
                            logger.info(self.history_messages)
                            
                            tool_usage_info.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "name": item.name,
                                "args": item.arguments,
                                "output": json.dumps({result_name: tool_call_result})
                            })
                            logger.info(f"\n--- function calling ---:  {item.name} \n {tool_call_result}\n")
                        else:
                            print(response.output)
                            raise KeyError("Doctor Agent abnomarl chat.")  