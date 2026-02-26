# image_prompt_generator.py

import os
import re
from typing import List, Dict
from langchain_openai import ChatOpenAI

class ImagePromptGenerator:
    def __init__(self):
        """Initializes the generator with a creative LLM."""
        self.llm = ChatOpenAI(
            model="sarvam-m",
            temperature=0.7,
            base_url="https://api.sarvam.ai/v1",
            api_key=os.getenv("SARVAM_API_KEY"),
            default_headers={"api-subscription-key": os.getenv("SARVAM_API_KEY")},
            stream_usage=False,
        )

    def _parse_report(self, report_content: str) -> List[Dict[str, str]]:
        """Parses the final report to extract individual source analyses."""
        # This regex matches '### **Title**' and captures the title and its content until the next '###' or end of string
        pattern = r"### \*\*(.*?)\*\*\s*\n(.*?)(?=\n### |\Z)"
        matches = re.findall(pattern, report_content, re.DOTALL)
        
        sources = []
        for title, content in matches:
            sources.append({
                "title": title.strip(),
                "content": content.strip()
            })
        return sources

    def _create_prompts_for_source(self, source_analysis: str) -> str:
        """Makes a single LLM call to generate six detailed image prompts for one source."""
        generation_prompt = f"""
        You are a creative director and concept artist for a tech podcast. Your task is to generate 6 distinct, highly-detailed text-to-image prompts based on the provided analysis of a source. Refer to different aspects of the source to create diverse visual concepts.

        Source Analysis:
        ---
        {source_analysis}
        ---

        Instructions:
        1. Read the analysis to understand its core theme and argument.
        2. Create six different visual concepts that capture the essence of this theme.
        3. For each concept, write a detailed, 50-60 word prompt suitable for an advanced AI image generator (like Midjourney or DALL-E 3).
        4. Describe the scene, subject, mood, and style. Use powerful keywords. Be evocative and specific.
        5. Do not write any extra text or explanations. Output ONLY the six prompts.

        Format:
        Prompt 1: [Your first detailed 50-60 word image prompt here.]
        Prompt 2: [Your second detailed 50-60 word image prompt here.]
        Prompt 3: [Your third detailed 50-60 word image prompt here.]
        Prompt 4: [Your fourth detailed 50-60 word image prompt here.]
        Prompt 5: [Your fifth detailed 50-60 word image prompt here.]
        Prompt 6: [Your sixth detailed 50-60 word image prompt here.]
        """
        
        try:
            response = self.llm.invoke(generation_prompt)
            return response.content
        except Exception as e:
            print(f"Error generating image prompts: {e}")
            return "Prompt 1: Error in generation.\nPrompt 2: Error in generation."

    def generate_and_save_prompts(self, report_content: str, output_filename: str):
        """Main method to parse a report, generate prompts, and save them."""
        print("\n🎨 Starting image prompt generation...")
        sources = report_content
        
        print(f"\ncontent:\n {report_content}")
        print(f"\nsources:\n {sources}")
        

        if not sources:
            print("Could not parse any sources from the report to generate prompts for.")
            return

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("# AI-Generated Image Prompts\n\n")
            f.write("Based on the final research report.\n\n")

            print(f"🖼️  Generating prompts for source:")
            f.write(f"## Prompts for Source:\n\n")
            
            prompts_text = self._create_prompts_for_source(sources)
            f.write(prompts_text)
            f.write("\n\n")
        
        print(f"✅ Image prompts successfully saved to: {output_filename}")

