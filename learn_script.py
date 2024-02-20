import os
import learn
import sys
import shutil

defaultFolder = "benchmarks-main/"
problemsFolder = "/training/easy/"

def main():
    domainFolders = [f.name for f in os.scandir(defaultFolder)]
    for domain in domainFolders:
        if  domain != "satellite" and domain != "blocksworld":
            continue
        if os.path.isdir(defaultFolder + domain):
            print("Running learning for: " + domain)
            #copy domain file to problemfolder
            if not os.path.exists(defaultFolder + domain + problemsFolder +"domain.pddl"):
                shutil.copy(defaultFolder + domain +"/domain.pddl",
                            defaultFolder + domain + problemsFolder +"domain.pddl")

            sys.argv.clear()
            sys.argv.append(defaultFolder + domain)
            sys.argv.append(defaultFolder + domain + problemsFolder)
            learn.main()
            if os.path.exists("training-result/" + domain):
                shutil.rmtree("training-result/" + domain)
            shutil.copytree("data/partial-grounding-bad-rules", "training-result/" + domain + "/bad-rules")
            shutil.copytree("data/partial-grounding-good-rules", "training-result/" + domain + "/good-rules")
            



if __name__ == "__main__":
    main()