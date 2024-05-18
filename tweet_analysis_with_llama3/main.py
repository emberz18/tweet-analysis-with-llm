import json

from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.llms import BaseLLM

from tweet_analysis_with_llama3.template.analysis import analysis_template


def setup_llm() -> BaseLLM:
    return Ollama(model="llama3")


def setup_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(template=analysis_template)


def parse_tweets():
    with open("../tweet_histories/tweets.json") as f:
        tweets = json.load(f)

    return [
        f"{tweet['tweet']['edit_info']['initial']['editableUntil']}: {tweet['tweet']['full_text']}"
        for tweet in tweets
        if not "RT" in tweet["tweet"]["full_text"]
    ]


def main():
    llm = setup_llm()
    template = setup_prompt()
    chain = template | llm 

    tweets = parse_tweets()

    response = chain.invoke({"tweets": "\n".join(tweets)})
    print(response)


if __name__ == "__main__":
    main()
