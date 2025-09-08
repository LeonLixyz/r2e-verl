messages= [
    {
      "role": "system",
      "content": '''You are a skilled programmer specializing in competitive programming and algorithmic problem solving.
        Your task is to solve coding problems efficiently and correctly.

        General approach:
        1. Carefully analyze the problem statement and requirements
        2. Identify the appropriate algorithms and data structures needed
        3. Implement a clean, efficient solution
        4. Test the solution with various test cases including edge cases
        5. Optimize if necessary for time/space complexity

        Guidelines:
        - Write clean, well-commented code
        - Handle edge cases properly
        - Prefer optimal solutions but start with a working solution first
        - Use appropriate programming languages (Python, C++, Java, etc.)
        - Always test your code with sample inputs and edge cases
        - Explain your approach and time/space complexity
        - If the problem involves multiple solutions, implement the most efficient one
        - Debug systematically if tests fail
        
        Available tools:
        - edit_file: Create or modify code files
        - bash: Compile, run, and test your solutions
        - view_file: View the contents of a file or directory
        - delete_files: Delete files or directories
        
        Focus areas:
        - Data structures (arrays, trees, graphs, heaps, etc.)
        - Algorithms (sorting, searching, dynamic programming, greedy, etc.)
        - Competitive programming techniques
        - Time and space complexity optimization
        - Edge case handling
        - Clean code practices'''
    }
  ]


tools = [
   {
      "type": "function",
      "function": {
        "name": "view_file",
        "description": "Tool for viewing/exploring the contents of existing files\n* If `path` is a file, `view` displays the result of applying `cat -n`.\n* If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* If the output is too long, it will be truncated and marked so \n",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`."
            },
            "view_range": {
              "type": "array",
              "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
              "items": {
                "type": "integer"
              }
            }
          },
          "required": [
            "path"
          ]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "edit_file",
        "description": "Use this tool to propose an edit to an existing file or create a new file.\n\nThis will be read by a less intelligent model, which will quickly apply the edit. You should make it clear what the edit is, while also minimizing the unchanged code you write.\nWhen writing the edit, you should specify each edit in sequence, with the special comment `// ... existing code ...` to represent unchanged code in between edited lines.\n\nFor example:\n\n```\n// ... existing code ...\nFIRST_EDIT\n// ... existing code ...\nSECOND_EDIT\n// ... existing code ...\nTHIRD_EDIT\n// ... existing code ...\n```\n\nYou should still bias towards repeating as few lines of the original file as possible to convey the change.\nBut, each edit should contain sufficient context of unchanged lines around the code you're editing to resolve ambiguity.\nDO NOT omit spans of pre-existing code (or comments) without using the `// ... existing code ...` comment to indicate its absence. If you omit the existing code comment, the model may inadvertently delete these lines.\nMake sure it is clear what the edit should be, and where it should be applied.\nTo create a new file, simply specify the content of the file in the `code_edit` field.\n\nYou should specify the following arguments before the others: [target_file]",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "The target file to modify. Always specify the target file as the first argument. You must use an absolute path."
            },
            "instructions": {
              "type": "string",
              "description": "A single sentence instruction describing what you are going to do for the sketched edit. This is used to assist the less intelligent model in applying the edit. Please use the first person to describe what you are going to do. Dont repeat what you have said previously in normal messages. And use it to disambiguate uncertainty in the edit."
            },
            "code_edit": {
              "type": "string",
              "description": "Specify ONLY the precise lines of code that you wish to edit. **NEVER specify or write out unchanged code**. Instead, represent all unchanged code using the comment of the language you're editing in - example: `// ... existing code ...`"
            }
          },
          "required": [
            "path",
            "instructions",
            "code_edit"
          ]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "delete_files",
        "description": "Deletes multiple files or directories at the specified paths. Each operation will fail gracefully if:\n* The file doesn't exist\n* The file cannot be deleted",
        "parameters": {
          "type": "object",
          "properties": {
            "target_file_paths": {
              "description": "Array of file or directory paths to delete",
              "items": {
                "type": "string"
              },
              "type": "array"
            }
          },
          "required": [
            "target_file_paths"
          ]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Run commands in a bash shell\n* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n* You have access to a mirror of common linux and python packages via apt and pip.\n* State is persistent across command calls and discussions with the user.\n* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n* Please avoid commands that may produce a very large amount of output.\n* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string",
              "description": "The bash command to run. Required unless the tool is being restarted."
            },
            "restart": {
              "type": "boolean",
              "description": "Specifying true will restart this tool. Otherwise, leave this unspecified."
            }
          }
        }
      }
    }
]


