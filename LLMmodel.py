#用作测试实例的智谱GLM大模型
from langchain.llms.base import LLM
from zhipuai import ZhipuAI
from langchain_core.messages import AIMessage ,HumanMessage
from typing import List,Dict
class ChatGLM4(LLM):
    client: object = None
    def __init__(self):
        super().__init__()
        self.client = ZhipuAI(api_key="a542f179df2831b7762f1c1cd251ff8d.NepN5i3RVd1TdkbS")#密钥后面可以放在环境配置里面，让os自动查找填入

    def _llm_type(self):
        return "ChatGLM4"

    def invoke(self, prompt, history=[] or None):#简单对话函数
        if history is None:
            history=[]
        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model="glm-4", messages=history)
        result = response.choices[0].message.content
        print(result)
        return AIMessage(content=result)

    def invokeType(self, prompt, history=[] or None):  # 简单对话函数
        if history is None:
            history = []
        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model="glm-4", messages=history)
        result = response.choices[0].message.content
        print(result)
        return result

    def _call(self, prompt, history=[]):#自动调用
        return self.invoke(prompt, history)

    def stream(self, prompt, history=[]): #流式输出
        if history is None:
            history = []

        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model="glm-4", messages=history, stream=True)
        for chunk in response:
           print ( chunk.choices[0].delta.content,end='')


#这个函数也可以单独拆出来放在一个独立文件夹里面
    def construct_Voicemessages(self,question, history: List[List | None],prompt=str) -> List[Dict[str, str]]:
      messages = [
        {"role": "system",
         "content": "你现在扮演用户意图识别的角色，要求根据用户输入和AI的回答，正确提取出信息，无需包含提示文字"}]
      for entry in history:
        if entry["role"] == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "assistant":
            messages.append({"role": "assistant", "content": entry["content"]})
      messages.append({"role": "user", "content": question})
      messages.append({"role": "user", "content": prompt})
      return messages
    
if __name__=='__main__':
     #测试
     mybot=ChatGLM4()
     mybot.invoke("番茄炒蛋的做法是什么?")
     mybot.stream("西瓜吃法有哪些？")

     """ prompt="请你从上述对话中，提取出用户即将想要进行语音输出的文本，不要提示信息，如果找不到，请输出None"
         history= [
                {"role":"user", "content":"番茄炒蛋的做法是什么，请用语音进行输出？"},
                {"role": "assistant","content":"起锅烧油，放入鸡蛋翻炒即可"},
                {"role":"user", "content":"如何做一杯拿铁咖啡？请用语音回答"},
                {"role": "assistant", "content":"首先煮一壶热水，然后取适量咖啡粉，用热水冲泡，最后加入适量的牛奶即可。尽管我目前不能语音输出，但是我仍在努力优化"},
                {"role":"user", "content":"如何在家做披萨？"},
                {"role": "assistant", "content":"可以购买披萨饼底，然后添加你喜欢的配料和芝士，放入烤箱烤至金黄即可"},       
                   ] 
         historymessage = mybot.construct_Voicemessages("夏天如何穿搭配？",history=history,prompt=prompt)
          mybot.invoke (prompt="夏天应该吃什么水果?" ,history=historymessage)
     """