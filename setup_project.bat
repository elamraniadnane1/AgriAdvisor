@echo off
REM Create main project directory
mkdir agri-project
cd agri-project

REM Create ReactJS frontend directory and subdirectories
mkdir agri-app
cd agri-app
mkdir node_modules
mkdir public
mkdir src
cd src
mkdir components
cd components
echo. > Register.js
echo. > Login.js
echo. > Query.js
echo. > Logout.js
cd ..
echo. > App.js
echo. > index.js
echo. > App.css
echo. > index.css
cd ..
echo. > .gitignore
echo. > package.json
echo. > package-lock.json
echo. > README.md
echo. > .env
cd ..

REM Create ExpressJS backend directory and subdirectories
mkdir backend
cd backend
mkdir node_modules
mkdir scripts
cd scripts
echo. > process_pdfs.py
echo. > vectorize_and_store.py
echo. > generate_response.py
echo. > text_to_speech.py
cd ..
echo. > server.js
echo. > package.json
echo. > package-lock.json
echo. > .gitignore
echo. > README.md
cd ..

REM Create main project files
echo. > .gitignore
echo. > README.md

echo Project structure created successfully!
pause
