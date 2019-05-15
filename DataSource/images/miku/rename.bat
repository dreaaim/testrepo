@echo off
set a = 0
setlocal EnableDelayedExpansion
for %%n in (*.JPG) do (
set /A a+=1
ren "%%n" "miku_!a!.jpg"
)