

from guardrails import Guard, OnFailAction
from guardrails.hub import BanList, BiasCheck, NSFWText, ProfanityFree, LogicCheck, MentionsDrugs, PolitenessCheck, ToxicLanguage#, ToxicLanguage# Updated import
#import guardrails.hub #which needs fuzzysearch, which is already downloaded
#from guardrails.datatypes import String
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from guardrails import Guard


guard = Guard().use_many(
    BiasCheck(
        threshold=0.5,
        on_fail="fix"
    ),

    NSFWText(
        threshold=0.8,
        validation_method="sentence"
    ),

    ProfanityFree(
        on_fail = "fix"
    ),

    LogicCheck(
        model="gpt-3.5-turbo",
        on_fail="fix"
    ),

    MentionsDrugs(
        on_fail = "fix"
    ),

    PolitenessCheck(
        llm_callable="gpt-3.5-turbo",
        on_fail = "fix"
    ),

    ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        on_fail="fix"
    )

)

print(guard)

class Test:

    def guardrails(self, input):
        #if guardrails return true send back whatever the input is,
        #else send back an error message
        try:
            guard.validate(input)
            return True
        except Exception as e:
            print("Before e")
            print(e)
            print("testing")
            return False
    def test(self):
        print("Result: " + str(self.guardrails("What is inheritance?")))


test = Test()
test.test()