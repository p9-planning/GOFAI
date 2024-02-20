import os
import learn
import sys
import shutil

defaultFolder = "training-result/"


def main():
    domainFolders = [f.name for f in os.scandir(defaultFolder)]
    for domain in domainFolders:
        if  domain != "satellite" and domain != "blocksworld":
            continue
        if os.path.isdir(defaultFolder + domain):
            print("Running learning for: " + domain)
            sys.argv.clear()
            sys.argv.append("domain_knowledge", )#path to domain knowledge file
            sys.argv.append("domain", )#path to domain file
            sys.argv.append("problem", )#path to problem file
            sys.argv.append("plan", )#path to output plan file
            learn.main()
            if os.path.exists("training-result/" + domain):
                shutil.rmtree("training-result/" + domain)
            shutil.copytree("data/partial-grounding-bad-rules", "training-result/" + domain + "/bad-rules")
            shutil.copytree("data/partial-grounding-good-rules", "training-result/" + domain + "/good-rules")
            



if __name__ == "__main__":
    main()