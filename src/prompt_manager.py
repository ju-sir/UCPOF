from typing import Dict, List, Optional, Any

class PromptManager:
    def __init__(self, dataset_config: Dict[str, Any]):
        """初始化Prompt管理器"""
        self.dataset_config = dataset_config
        self.templates: Dict[str, str] = dataset_config.get('templates', {})
    
    def build_prompt(self, input_text: str, rag_context: Optional[str] = None, prompt_type: str = "examples_task") -> List[Dict[str, str]]:
        """构建动态Prompt"""
        if prompt_type == "examples_task":
            # Examples -> Task 顺序
            system_content = self.templates.get('good_examples', '') + "\n" + self.templates.get('task', '')
        elif prompt_type == "task_examples":
            # Task -> Examples 顺序
            system_content = self.templates.get('task', '') + "\n" + self.templates.get('good_examples', '')
        else:
            # 默认顺序
            system_content = self.templates.get('good_examples', '') + "\n" + self.templates.get('task', '')
        
        # 如果有RAG上下文，添加到system_content中
        if rag_context:
            system_content += "\n\n--- RELEVANT CONTEXT ---\n" + rag_context
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Input: {input_text}\nOutput: "}
        ]
        
        return messages
    
    def get_chat_template(self, messages: List[Dict[str, str]], tokenizer: Any) -> str:
        """获取聊天模板"""
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
