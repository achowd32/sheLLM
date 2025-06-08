import language_tool_python
import sys

def main(text):
    tool = language_tool_python.LanguageTool('en-US')
    errors = tool.check(text)
    print(len(errors))

if __name__=="__main__":
    main(sys.argv[1])