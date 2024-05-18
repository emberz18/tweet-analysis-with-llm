import json
import re

import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.llms import BaseLLM

from tweet_analysis_with_llama3.template.analysis import analysis_template


def setup_llm() -> BaseLLM:
    return Ollama(model="llama3")


def setup_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(template=analysis_template)


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


@cl.on_chat_start
async def start():
    llm = setup_llm()
    template = setup_prompt()
    chain = template | llm

    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="ファイルをアップロード次第ツイートの内容を分析します。",
            accept={"text/plain": [".js"]},
            max_size_mb=200,
        ).send()

    tweets = parse_tweets(files[0].path)

    response = await chain.ainvoke(
        "\n".join(tweets), callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=response).send()
