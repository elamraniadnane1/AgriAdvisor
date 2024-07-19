@echo off

:: Create the main project directory
mkdir my_project

:: Change to the main project directory
cd my_project

:: Create the main Python script
echo. > main.py

:: Create the JSON file for users
echo. > users.json

:: Create the templates directory and its files
mkdir templates
cd templates
echo. > base.html
echo. > index.html
echo. > login.html
echo. > register.html

:: Go back to the main project directory
cd ..

:: Create the static directory and its files
mkdir static
cd static
echo. > styles.css
echo. > script.js

:: Go back to the main project directory
cd ..

echo Directory structure and files created successfully.
