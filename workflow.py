# multi_agent_workflow.py
import re
from dotenv import load_dotenv
from typing import Dict, Any, List, TypedDict, Annotated
import operator
from time import sleep

from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilyExtract, TavilySearch
from image_prompt import ImagePromptGenerator

# Load environment variables
load_dotenv()

# --- 1. State Definition for the Core Research Loop (Unchanged) ---
class ResearchState(TypedDict):
    topic: str
    iteration: int
    plan: str
    search_queries: List[str]
    raw_search_results: Annotated[List[dict], operator.add]
    extracted_pages: Annotated[List[dict], operator.add]
    draft_report: str
    critique: str
    research_history: Annotated[List[str], operator.add]
    final_report: str


# --- 2. NEW: The Topic Scout Agent ---
class TopicScout:
    """A specialized agent to find trending and viral topics."""
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.7)
        self.search_tool = TavilySearch(max_results=2)

    def find_trending_topic(self) -> str:
        """
        Executes a multi-step process to discover and synthesize a viral topic.
        """
        print("\n🕵️  Topic Scout is hunting for trends...")
        
        # Step 1: Search across different platforms with targeted queries
        print("  - Scanning Reddit and X/Twitter for viral discussions...")
        try:
            reddit_trends = self.search_tool.invoke({"query": "trending topics on reddit this week"})
            twitter_trends = self.search_tool.invoke({"query": "top viral discussions and hashtags on X (formerly Twitter) this week"})
        except Exception as e:
            print(f"Search failed during trend discovery: {e}")
            return "Error: Could not fetch trends. Please provide a topic manually."

        # Step 2: Combine all findings into a single context for analysis
        combined_context = "Reddit Trends:\n" + "\n".join([str(t) for t in reddit_trends])
        combined_context += "\n\nTwitter/X Trends:\n" + "\n".join([str(t) for t in twitter_trends])

        # Step 3: Use an LLM to analyze the raw data and synthesize a topic
        print("  - Analyzing trends to identify a core topic...")
        synthesis_prompt = f"""
        You are a Viral Trend Analyst. You have been given raw data from social media feeds. Your task is to analyze this data to identify the most engaging, viral, and intellectually stimulating topic.

        Instructions:
        1. Read through all the provided search results from Reddit and Twitter/X.
        2. Identify the recurring themes, keywords, and questions that are generating the most buzz.
        3. Distill these findings into a single, compelling, and research-worthy topic.
        4. The topic should be formulated as a question or a statement that can be deeply investigated.
        5. Output ONLY the final topic string and nothing else.

        Raw Data:
        ---
        {combined_context}
        ---

        Final Topic:
        """
        
        try:
            response = self.llm.invoke(synthesis_prompt)
            final_topic = response.content.strip()
            return final_topic
        except Exception as e:
            print(f"Topic synthesis failed: {e}")
            return "Error: Could not synthesize a topic. Please provide one manually."


# --- 3. The Main Multi-Agent Research System (Unchanged Core Logic) ---
class MultiAgentResearchSystem:
    def __init__(self):
        self.planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)
        self.writer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.5)
        self.judge_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        self.search_tool = TavilySearch(max_results=2, topic="general", search_depth="basic")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    # --- Agent Node Definitions (All Unchanged) ---
    def _planner_node(self, state: ResearchState) -> Dict[str, Any]:
        iteration = state['iteration']
        print(f"\n--- Iteration {iteration}: Planner Agent is thinking... ---")
        if iteration == 1:
            history_prompt = "This is the first iteration. Start a fresh research plan."
        else:
            history_prompt = (
                "You are in a subsequent iteration. Review the research history and the judge's critique "
                "to refine the plan. Do NOT repeat previous searches. Focus on addressing the critique and "
                "filling the identified knowledge gaps."
            )
        prompt = f"""
        You are a master research planner. Your role is to devise a detailed research strategy for the given topic.
        Topic: "{state['topic']}"
        Research History:\n{state['research_history']}
        Previous Critique (from the Judge):\n{state['critique']}
        Your Task ({history_prompt}):
        1. Create a concise, one-paragraph research plan for this iteration.
        2. Based on the plan, generate a list of 2 precise and diverse search queries to execute.
        Output Format:
        Plan:
        [Your one-paragraph research plan here.]
        Queries:
        - [Query 1]
        - [Query 2]
        """
        response = self.planner_llm.invoke(prompt)
        plan_match = re.search(r"Plan:\s*(.*?)\s*Queries:", response.content, re.DOTALL)
        queries_match = re.search(r"Queries:\s*(.*)", response.content, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else "No plan generated."
        queries = [q.strip().replace('-', '').strip() for q in queries_match.group(1).strip().split('\n')] if queries_match else []
        return {"plan": plan, "search_queries": queries, "raw_search_results": []}

    # --- IMPROVED Searcher Node ---
    def _searcher_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Executes the queries from the Planner with enhanced reliability and logging.
        """
        print("--- Searcher Agent is executing queries... ---")
        all_results = []
        # Filter out any empty strings that might have been parsed from the planner's output
        valid_queries = [q.strip("\"'") for q in state['search_queries'] if q]
        print(f"valid_queries are {valid_queries}")

        for query in valid_queries:
            print(f"  - Searching for: {query}")
            sleep(2)
            try:
                results = self.search_tool.invoke({"query": query})
                print(f"\n{results['results']}\n")
                if results: # Only extend if results are not empty
                    all_results.extend(results['results'])
                    print(f"    -> Found {len(results)} result(s).")
                else:
                    print("    -> No results returned for this query.")
            except Exception as e:
                print(f"    -> Search failed for query '{query}': {e}")
        
        print(f"--- Searcher Agent finished. Total results gathered: {len(all_results)} ---")
        return {"raw_search_results": all_results}
    
    def _extractor_node(self, state: ResearchState) -> dict:
        urls = [result["url"] for result in state["raw_search_results"]]
        urls = urls[-4:]

        """
        if state["iteration"] == 1:
            with open("urls.txt", "w", encoding="utf-8") as f:
                for url in urls:
                    f.write(url + "\n")
        else:
            with open("urls.txt", "a", encoding="utf-8") as f:
                for url in urls:
                    f.write(url + "\n")
        """

        print(f"{len(urls)} URLs found for extraction: {urls}")
        if not urls:
            print("No URLs")
            return {"extracted_pages": []}

        extractor = TavilyExtract(extract_depth="advanced", format="markdown", inlude_images=False)
        extract_resp = extractor.invoke({"urls": urls})
        #saving extract_resp exactly as is in txt file
        #with open("extracted_pages.txt", "w", encoding="utf-8") as f:
        #    f.write(str(extract_resp))
        print(f"Extractor pulled {len(extract_resp)} full page(s).")
        return {"extracted_pages": extract_resp["results"]}

    
    def _writer_node(self, state: ResearchState) -> Dict[str, Any]:
        print("--- Writer Agent is drafting the report... ---")
        sleep(10)

        if not state["extracted_pages"]:
            return {"draft_report": "No extracted pages were available for this iteration."}

        source_material = "\n\n---\n\n".join(
            page.get("raw_content", "") for page in state["extracted_pages"]
        )

        prompt = f"""
        You are an expert writer. Synthesise the following material into a detailed,
        well-structured report.

        Topic: "{state['topic']}"

        Instructions:
        - Analyze all the provided source material.
        - Write an in-depth report that covers the key findings, arguments, evidence, viewpoints.
        - Structure the report properly.
        - Integrate it into a coherent narrative for a podcast.
        - The report should be neutral.

        Source Material:
        {source_material}
        """
        response = self.writer_llm.invoke(prompt)
        return {"draft_report": response.content}


    def _judge_node(self, state: ResearchState) -> Dict[str, Any]:
        print("--- Judge Agent is critiquing the report... ---")
        sleep(20)
        prompt = f"""
        You are a highly critical and insightful judge. Your task is to evaluate a research report.
        Topic: "{state['topic']}"
        Your Persona: First, deduce the primary audience for this topic. Embody that persona for your critique.
        Report to Critique:
        {state['draft_report']}
        Your Task:
        1. Provide a concise, constructive critique of the report from your persona's point of view.
        2. Identify specific weaknesses, gaps in logic, or unanswered questions.
        3. Suggest 2-3 new, specific research angles or questions that the next research iteration should focus on to improve the report.
        Output Format:
        Critique:
        [Your detailed critique here.]
        Suggestions for Next Iteration:
        - [Suggestion/Question 1]
        """
        response = self.judge_llm.invoke(prompt)
        history_entry = (
            f"--- Iteration {state['iteration']} ---\n"
            f"Plan: {state['plan']}\n"
            f"Critique:\n{response.content}"
        )
        return {"critique": response.content, "research_history": [history_entry], "iteration": state['iteration'] + 1}
        
    def _final_writer_node(self, state: ResearchState) -> Dict[str, Any]:
        sleep(10)
        print("--- Final Report Compilation by Lead Editor... ---")
        full_context = (
            f"Topic: {state['topic']}\n\n"
            "Research History and Critiques:\n" + "\n".join(state['research_history']) +
            "\n\nFinal Draft Report to Polish:\n" + state['draft_report']
        )
        prompt = f"""
        You are the lead editor. Your task is to produce the final, polished version of a research report by integrating all context and the latest draft.
        Full Context:
        {full_context}
        Generate a final report based on this context, well structured as a discussion covering different viewpoints mentioned in it.
        """
        response = self.writer_llm.invoke(prompt)
        print(f"\nThis is the full context:\n{full_context}\n")
        return {"final_report": response.content}

    def _should_continue(self, state: ResearchState) -> str:
        if state['iteration'] > 2:
            return "end_workflow"
        return "continue_to_judge"

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("searcher", self._searcher_node)
        workflow.add_node("extractor",    self._extractor_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("final_writer", self._final_writer_node)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "searcher")
        workflow.add_edge("searcher", "extractor")
        workflow.add_edge("extractor", "writer")
        workflow.add_conditional_edges("writer", self._should_continue, {"continue_to_judge": "judge", "end_workflow": "final_writer"})
        workflow.add_edge("judge", "planner")
        workflow.add_edge("final_writer", END)
        return workflow

    # --- MODIFIED Public method to include Topic Scout ---

    def run(self, topic: str = None):
        """Runs the entire process, starting with user choice for topic discovery."""
        
        # Step 1: Determine the topic
        if not topic:
            while True:
                choice = input("Choose an option:\n1. Enter a topic manually\n2. Let the AI find a trending topic\nYour choice: ").strip()
                if choice in ['1', '2']:
                    break
                print("Invalid choice. Please enter 1 or 2.")
            
            if choice == '1':
                topic = input("Please enter the research topic: ").strip()
            else:
                scout = TopicScout()
                topic = scout.find_trending_topic()
                if "Error:" in topic:
                    print(topic) # Print the error message
                    return # Exit if topic discovery fails
                print(f"\n🤖 AI has selected the topic: \"{topic}\"")

        # Step 2: Run the research workflow with the determined topic
        initial_state = {
            "topic": topic, "iteration": 1, "plan": "", "search_queries": [],
            "raw_search_results": [], "draft_report": "", "critique": "",
            "research_history": [], "final_report": ""
        }
        
        final_state = None
        for output in self.app.stream(initial_state):
            for node_name, state_after_node in output.items():
                print(f"\n--- Output from: {node_name} ---")
                final_state = state_after_node

                if node_name == "planner":
                    print(f"Plan: {state_after_node['plan']}")
                    print(f"Queries: {state_after_node['search_queries']}")
                elif node_name == "writer":
                    print("Draft Report Generated (showing first 200 chars):")
                    print(state_after_node['draft_report'][:200])
                elif node_name == "judge":
                    print("\n**Critique and Suggestions from the Judge:**")
                    print(state_after_node['critique'])
        
        
        # --- Step 3: Save the final report ---
        final_report_content = final_state.get('final_report', "No report was generated.")
        report_filename = "final_research_report_" + re.sub(r'[^\w\s-]', '', topic.lower()).replace(' ', '_')[:50] + ".md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(final_report_content)
        
        print(f"\n Workflow complete! Final report saved to '{report_filename}'")
        
        """
        # --- NEW: Step 4: Generate and save image prompts ---
        if final_report_content and "No report was generated." not in final_report_content:
            image_gen = ImagePromptGenerator()
            image_prompt_filename = "image_prompts_for_" + re.sub(r'[^\w\s-]', '', topic.lower()).replace(' ', '_')[:50] + ".md"
            image_gen.generate_and_save_prompts(final_report_content, image_prompt_filename)
        else:
            print("\nSkipping image prompt generation as the final report was not successfully created.")
        """


        return final_report_content

# --- Main execution block ---
if __name__ == "__main__":
    research_system = MultiAgentResearchSystem()
    research_system.run()