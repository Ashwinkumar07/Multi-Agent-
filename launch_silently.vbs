Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "A:\Edufi_project"
WshShell.Run "python app.py", 0, False
