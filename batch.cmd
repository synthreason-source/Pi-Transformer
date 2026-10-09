@echo off
setlocal enabledelayedexpansion

set "ROOT=%CD%"
set "MUTATOR=%ROOT%\mutation_tester.py"
set "PROMPT=once upon a"

if not exist "%MUTATOR%" (
    echo Missing mutator: "%MUTATOR%"
    exit /b 1
)

for %%F in ("%ROOT%\*.py") do (
    if /I not "%%~nxF"=="markov_line_fuzzer.py" (
        echo Running mutator on %%~nxF
        python "%MUTATOR%" "%ROOT%\%%~nxF" xaa --prompt "%PROMPT%"
    )
)

endlocal
