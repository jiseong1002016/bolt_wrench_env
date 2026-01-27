import os
import re
from typing import List
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def write_urdf(modules : List, name : str = "robot", template_path : str = None) -> None:
    """
    Replace placeholders in a template file with contents from other files.    
    """    # Use default template path if none provided
    if template_path is None:
        template_path = os.path.join(SCRIPT_DIR, "_template")
    try:
        # Read the template file
        with open(template_path, 'r', encoding='utf-8') as template_file:
            urdf_string = template_file.read()
        
        # Replace each placeholder with content from corresponding file
        content = f"  <!-- [URDF-WRITER] Created with urdf-writer script (by CHH) on {datetime.now()}-->\n"
        merged_constraint_bodies: List[str] = []
        merged_nominal_tokens: List[str] = []

        constraints_pattern = re.compile(
            r"<constraints\b([^>]*)>(.*?)</constraints>",
            re.DOTALL,
        )

        for module_name in modules:
            file_path = os.path.join(SCRIPT_DIR, "..", "modules", module_name + ".xml")
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Warning: File '{file_path}' not found. Skipping...")
                continue
            
            try:
                # Read the module file
                with open(file_path, 'r', encoding='utf-8') as module_file:
                    module_text = module_file.read()

                # Extract and remove any <constraints> blocks from the module.
                def _extract_constraints(match: re.Match) -> str:
                    attrs = match.group(1) or ""
                    body = (match.group(2) or "").strip()

                    nominal_match = re.search(
                        r'nominal_config\s*=\s*"([^"]*)"',
                        attrs,
                    )
                    if nominal_match:
                        nominal_values = nominal_match.group(1).strip()
                        if nominal_values:
                            merged_nominal_tokens.extend(nominal_values.split())

                    if body:
                        merged_constraint_bodies.append(body)

                    # Remove this constraints block from the module content.
                    return ""

                module_text = constraints_pattern.sub(_extract_constraints, module_text)

                content += (
                    f"\n  <!-- [URDF-WRITER] from module [{module_name}] -->\n"
                    + module_text
                )
                
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}")
                continue

        # Append a single merged <constraints> block at the end, if any were found.
        if merged_constraint_bodies:
            nominal_attr = ""
            if merged_nominal_tokens:
                nominal_attr = f' nominal_config="{" ".join(merged_nominal_tokens)}"'

            merged_body = "\n".join(f"    {body}" for body in merged_constraint_bodies)
            content += (
                "\n  <!-- [URDF-WRITER] merged constraints -->\n"
                f"  <constraints{nominal_attr}>\n"
                f"{merged_body}\n"
                "  </constraints>\n"
            )
        
        # Save the processed template to output file

        urdf_string = urdf_string.replace("$NAME$",name).replace("$CONTENT$",content)

        output_path = os.path.join(SCRIPT_DIR, "..", "urdf", f"{name}.urdf")
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(urdf_string)
        
        print(f"[{name}] written to '{output_path}'")
        
    except FileNotFoundError:
        print(f"Error: Template file '{template_path}' not found.")
    except Exception as e:
        print(f"Error processing template: {e}")

# Example usage
if __name__ == "__main__":
    write_urdf(["_fixed-base", "raipal_L"],"raipal_L")
    write_urdf(["_fixed-base", "raipal_R"],"raipal_R")

    write_urdf(["_fixed-base", "raipal-upper_L"],"raipal_upper-only_L")
    write_urdf(["_fixed-base", "raipal-upper_R"],"raipal_upper-only_R")

    write_urdf(["_fixed-base", "raipal_L", "saber_L"],"raipal_saber_L")
    write_urdf(["_fixed-base", "raipal_R", "saber_R"],"raipal_saber_R")

    write_urdf(["_fixed-base", "raipal_L", "stub-10_L"],"raipal_stub-10_L")
    write_urdf(["_fixed-base", "raipal_L", "stub-5_L" ],"raipal_stub-5_L")
    write_urdf(["_fixed-base", "raipal_L", "stub-0_L" ],"raipal_stub-0_L")

    write_urdf(["_fixed-base", "raipal_R", "stub-10_R"],"raipal_stub-10_R")
    write_urdf(["_fixed-base", "raipal_R", "stub-5_R" ],"raipal_stub-5_R")
    write_urdf(["_fixed-base", "raipal_R", "stub-0_R" ],"raipal_stub-0_R")

    write_urdf(["_fixed-base", "raipal_R", "stub-0_R" ,"raipal_L","stub-0_L" ],"raipal_stub-0")
    write_urdf(["_fixed-base", "raipal_R", "stub-5_R" ,"raipal_L","stub-5_L" ],"raipal_stub-5")
    write_urdf(["_fixed-base", "raipal_R", "stub-10_R","raipal_L","stub-10_L"],"raipal_stub-10")

    write_urdf(["_fixed-base", "raipal_L", "end-effector"],"raipal_end-effector_L")
