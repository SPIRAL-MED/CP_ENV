import jsonlines
from rich import print
from openai import OpenAI
from dataclasses import dataclass, field


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.tool_manager import ToolManager
    from utils.agents import PatientAgent


@dataclass
class RunContext:
    """
    Data container responsible for carrying all runtime data.
    Maintains the state context for the patient-doctor simulation loop.
    """
    # --- Source Data ---
    data_item: dict
    output_path: str
    
    # --- Internal Agents & Tools ---
    patient_client: OpenAI = field(init=False)
    doctor_client: OpenAI = field(init=False)
    patient_agent: "PatientAgent" = field(init=False)
    tool_manager: "ToolManager" = field(init=False)
    
    # --- Agent Result Data (Simulation Outcomes) ---
    process_success: bool = field(init=False)
    patient_records: list = field(init=False)
    patient_records_simplified: list = field(init=False)
    diagnosis_reports: list = field(init=False)
    tool_usages: list = field(init=False)
    doctor_patient_diaglogues: list = field(init=False)
    
    def output_agent_results(self):
        """
        Aggregates the simulation results and appends them to the output JSONL file.
        """
        results = {
            "original_data": self.data_item,
            "process_success": self.process_success,
            "patient_records": self.patient_records,
            "diagnosis_reports": self.diagnosis_reports,
            "tool_usages": self.tool_usages,
            "doctor_patient_diaglogues": self.doctor_patient_diaglogues
        }
        # print(results)
        with jsonlines.open(self.output_path, "a") as writer:
            writer.write(results)