@echo off
chcp 65001 >nul
title Démarrage du Générateur de Fiches de Cours Multi-Agents

echo ============================================================
echo 🚀 Démarrage du Générateur de Fiches de Cours Multi-Agents
echo ============================================================
echo.

:: Activer l'environnement virtuel
if exist ".venv\" (
    echo ✓ Activation de l'environnement virtuel...
    call .venv\Scripts\activate.bat
) else (
    echo ❌ Environnement virtuel .venv non trouvé!
    echo    Veuillez créer un environnement virtuel avec: python -m venv .venv
    pause
    exit /b 1
)

:: Vérifier si les dépendances sont installées
@echo off
echo ========================================
echo Generateur de Fiches de Cours Multi-Agents
echo ========================================
echo.

REM Verifier l'environnement virtuel
if not exist ".venv\" (
    echo [ERREUR] Environnement virtuel .venv non trouve!
    echo Veuillez creer un environnement virtuel avec: python -m venv .venv
    pause
    exit /b 1
)

echo [OK] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

REM Verifier si streamlit est installe
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [INFO] Installation des dependances...
    pip install -r requirements.txt
)

REM Creer les dossiers necessaires
echo [INFO] Verification des dossiers...
if not exist "Corpus\Informatique\" mkdir "Corpus\Informatique"
if not exist "Corpus\Mathematiques\" mkdir "Corpus\Mathematiques"
if not exist "vectorstore\" mkdir "vectorstore"
if not exist "output\" mkdir "output"

echo.
echo ========================================
echo Lancement de l'application Streamlit...
echo URL: http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.

streamlit run app.py

pause