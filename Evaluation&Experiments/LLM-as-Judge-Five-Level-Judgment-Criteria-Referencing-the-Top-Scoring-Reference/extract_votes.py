import json
import re


def extract_votes(filename, mode):
    with open(f"VERDICTS_{mode}_Qs/" + filename + "_VERDICTS.txt", 'r') as file:
        data = file.read()

    jsons = data.split("}{")
    queries = []
    

    votes = []
    for dic in jsons:
        query = dic.split('"query": ', 1)[1].split(', ')[0]
        queries.append(query)
        answer = dic.split('"answer": ', 1)[1]

        vote = answer.split("[RESULT]", 1)[1].replace('"', "").replace(" ", "")
        if "}" in vote:
            vote = vote.replace("}", "")
        if vote:
            #print(f"[+] The vote is : {vote}")
            votes.append(int(vote))
        else:
            print("No verdict found.")

    print(filename)
    summa = sum(votes)
    average = summa / len(votes)
    print(average)


folder = input("\nDo you want to extract: [1] Multi-Hop Qs, [2] General Qs. Select 1 or 2 ...\n")
files = ["MoRSE", "GPT4", "GEMINI", "HACKERGPT", "MIXTRAL"]
#files = ["MoRSE", "GPT4", "PERPLEXITY"] Only for CVE_Qs
if folder == 1:
    mode = "MultiHop"
elif folder == 2:
    mode = "General"
for filename in files:
    extract_votes(filename, mode)


