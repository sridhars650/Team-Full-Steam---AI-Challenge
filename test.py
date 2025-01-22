

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
        on_fail="exception"
    ),

    NSFWText(
        threshold=0.8,
        validation_method="sentence"
    ),

    ProfanityFree(
        on_fail = "exception"
    ),

    LogicCheck(
        model="gpt-3.5-turbo",
        on_fail="exception"
    ),

    MentionsDrugs(
        on_fail = "exception"
    ),

    PolitenessCheck(
        llm_callable="gpt-3.5-turbo",
        on_fail = "exception"
    ),

    ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        on_fail="exception"
    )

)

class Test:

    def guardrails(self, input):
        #if guardrails return true send back whatever the input is,
        #else send back an error message
        try:
            guard.validate(input)
            return True
        except Exception as e:
            print(e)
            return False
    def test(self):
        print("Result: " + self.guardrails("What is an EDR?"))


test = Test()
test.test()