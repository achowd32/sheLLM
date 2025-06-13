import language_tool_python
import sys

def main(text):
    tool = language_tool_python.LanguageTool('en-US')
    errors = tool.check(text)
    num_errors = len(errors)
    print(f"This text has {num_errors} errors")

if __name__=="__main__":
    main(sys.argv[1])
