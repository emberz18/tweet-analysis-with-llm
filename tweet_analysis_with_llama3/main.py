import asyncio
import json
import os
import re
from pathlib import Path

import chainlit as cl
from langchain_core.runnables.base import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.llms import BaseLLM
from langchain_text_splitters import CharacterTextSplitter


sem = asyncio.Semaphore(1)

def setup_llm() -> BaseLLM:
    return Ollama(model="llama3")


def setup_template(file_path: str) -> PromptTemplate:
    with open(file_path, mode="r") as f:
        template_str = f.readlines()
    
    return PromptTemplate.from_template(template="".join(template_str))


def parse_tweets(file_path: str):
    with open(file_path) as f:
        js_content = f.read()

    json_string = re.findall(r"window\.YTD\.tweets\.part0 = (\[.*\])", js_content, re.DOTALL)[0]
    tweets = json.loads(json_string)
    
    return [
        f"{tweet['tweet']['edit_info']['initial']['editableUntil']}: {tweet['tweet']['full_text']}"
        for tweet in tweets
        if not "RT" in tweet["tweet"]["full_text"]
    ]

async def generate_summarization(map_chain: RunnableSequence, splitted_tweet: str, step_name: str, message):
    async with sem:
        message.content = f"{step_name}を実施中です。"
        await message.update()
        response = map_chain.invoke({"docs": splitted_tweet})

    async with cl.Step(name=step_name) as step:
        step.output = response
        await step.update()
    return response

@cl.on_chat_start
async def start():
    llm = setup_llm()
    template =  setup_template(os.path.join(Path(__file__).parent, "template/analysis.txt"))
    map_prompt = setup_template(os.path.join(Path(__file__).parent,"template/summarization.txt"))
    map_chain = map_prompt | llm
    reduce_chain = template | llm

    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="ファイルをアップロード次第ツイートの内容を分析します。",
            accept={"text/plain": [".js"]},
            max_size_mb=200,
        ).send()

    message = cl.Message(content="")
    await message.send()

    tweets = parse_tweets(files[0].path)
    summarizations = tweets

    i = 1
    while True:
        splitted_summarization = CharacterTextSplitter(
            chunk_size=4000, chunk_overlap=0
        ).split_text("\n".join(summarizations))

        if len(splitted_summarization) == 1:
            break

        tasks = [
            generate_summarization(map_chain, splitted_tweet, f"要約{i}_{j}", message) 
            for j,splitted_tweet in enumerate(splitted_summarization)
        ]
        summarizations = await asyncio.gather(*tasks)

        i+=1

    message.content = reduce_chain.invoke({"docs": splitted_summarization[0]})
    await message.update()

