@echo off

if exist build (
    rmdir /s /q build
    if %errorlevel% neq 0 (
        echo failed rm build
        exit
    )
)
mkdir build
cd build
cmake ..
multitop.sln
exit