#!/bin/bash

echo "=== REVIEWGUARDIAN - TESTS AUTOMATISES ==="
echo

echo "Verification rapide..."
python3 quick_test.py
echo

echo "Test complet du systeme..."
python3 test_complete_system.py
echo

echo "Tests termines!"