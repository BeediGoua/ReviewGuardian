@echo off
echo === REVIEWGUARDIAN - TESTS AUTOMATISES ===
echo.

echo Verification rapide...
py quick_test.py
echo.

echo Test complet du systeme...
py test_complete_system.py
echo.

echo Tests termines!
pause