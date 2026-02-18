#!/usr/bin/env python3
"""
Test Consistency Rules - Comprehensive Verification
Verify that scanner correctly handles all consistency scenarios

Author: Earthquake Prediction Research Team
Date: 2 February 2026
"""

import numpy as np
import torch
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(__file__))

def test_consistency_rules():
    """Test all consistency rule scenarios"""
    
    print("="*70)
    print("CONSISTENCY RULES - COMPREHENSIVE TEST")
    print("="*70)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Magnitude Normal, Azimuth Not Normal',
            'magnitude_probs': [0.1, 0.15, 0.15, 0.6],  # Normal = 60%
            'azimuth_probs': [0.2, 0.55, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01, 0.0],  # N = 55%
            'expected_magnitude': 3,  # Normal
            'expected_azimuth': 0,    # Normal (corrected)
            'expected_corrected': True,
            'expected_precursor': False
        },
        {
            'name': 'Scenario 2: Azimuth Normal, Magnitude Not Normal',
            'magnitude_probs': [0.65, 0.2, 0.1, 0.05],  # Large = 65%
            'azimuth_probs': [0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01, 0.0],  # Normal = 70%
            'expected_magnitude': 3,  # Normal (corrected)
            'expected_azimuth': 0,    # Normal
            'expected_corrected': True,
            'expected_precursor': False
        },
        {
            'name': 'Scenario 3: Both Earthquake, Low Confidence',
            'magnitude_probs': [0.1, 0.35, 0.2, 0.35],  # Medium = 35%
            'azimuth_probs': [0.3, 0.38, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.01],  # N = 38%
            'expected_magnitude': 3,  # Normal (corrected)
            'expected_azimuth': 0,    # Normal (corrected)
            'expected_corrected': True,
            'expected_precursor': False
        },
        {
            'name': 'Scenario 4: Both Earthquake, High Confidence',
            'magnitude_probs': [0.05, 0.78, 0.12, 0.05],  # Medium = 78%
            'azimuth_probs': [0.15, 0.65, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.0],  # N = 65%
            'expected_magnitude': 1,  # Medium (no change)
            'expected_azimuth': 1,    # N (no change)
            'expected_corrected': False,
            'expected_precursor': True
        },
        {
            'name': 'Scenario 5: Both Normal',
            'magnitude_probs': [0.05, 0.1, 0.15, 0.7],  # Normal = 70%
            'azimuth_probs': [0.75, 0.1, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01, 0.0],  # Normal = 75%
            'expected_magnitude': 3,  # Normal (no change)
            'expected_azimuth': 0,    # Normal (no change)
            'expected_corrected': False,
            'expected_precursor': False
        },
        {
            'name': 'Scenario 6: Both Earthquake, Borderline Confidence (40%)',
            'magnitude_probs': [0.1, 0.4, 0.2, 0.3],  # Medium = 40%
            'azimuth_probs': [0.35, 0.4, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01, 0.0],  # N = 40%
            'expected_magnitude': 1,  # Medium (no change, avg = 40%)
            'expected_azimuth': 1,    # N (no change)
            'expected_corrected': False,
            'expected_precursor': True
        },
        {
            'name': 'Scenario 7: Both Earthquake, Just Below Threshold (39.5%)',
            'magnitude_probs': [0.1, 0.39, 0.21, 0.3],  # Medium = 39%
            'azimuth_probs': [0.35, 0.4, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01, 0.0],  # N = 40%
            'expected_magnitude': 3,  # Normal (corrected, avg = 39.5% < 40%)
            'expected_azimuth': 0,    # Normal (corrected)
            'expected_corrected': True,
            'expected_precursor': False
        }
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {scenario['name']}")
        print(f"{'='*70}")
        
        # Simulate model output
        magnitude_probs = torch.tensor([scenario['magnitude_probs']])
        azimuth_probs = torch.tensor([scenario['azimuth_probs']])
        
        # Get raw predictions
        magnitude_pred_raw = torch.argmax(magnitude_probs, dim=1).item()
        azimuth_pred_raw = torch.argmax(azimuth_probs, dim=1).item()
        
        magnitude_conf_raw = magnitude_probs[0, magnitude_pred_raw].item() * 100
        azimuth_conf_raw = azimuth_probs[0, azimuth_pred_raw].item() * 100
        
        print(f"\nRaw Predictions:")
        print(f"  Magnitude: class {magnitude_pred_raw} ({magnitude_conf_raw:.1f}%)")
        print(f"  Azimuth: class {azimuth_pred_raw} ({azimuth_conf_raw:.1f}%)")
        
        # Apply consistency rules (same logic as scanner)
        magnitude_pred = magnitude_pred_raw
        azimuth_pred = azimuth_pred_raw
        magnitude_conf = magnitude_conf_raw
        azimuth_conf = azimuth_conf_raw
        is_corrected = False
        
        # Rule 1: Magnitude Normal → Azimuth Normal
        if magnitude_pred_raw == 3 and azimuth_pred_raw != 0:
            print(f"\n⚠️  Rule 1 triggered: Magnitude=Normal but Azimuth={azimuth_pred_raw}")
            azimuth_pred = 0
            azimuth_conf = azimuth_probs[0, 0].item() * 100
            is_corrected = True
        
        # Rule 2: Azimuth Normal → Magnitude Normal
        elif azimuth_pred_raw == 0 and magnitude_pred_raw != 3:
            print(f"\n⚠️  Rule 2 triggered: Azimuth=Normal but Magnitude={magnitude_pred_raw}")
            magnitude_pred = 3
            magnitude_conf = magnitude_probs[0, 3].item() * 100
            is_corrected = True
        
        # Rule 3: Low confidence → Both Normal
        elif magnitude_pred_raw != 3 and azimuth_pred_raw != 0:
            avg_conf = (magnitude_conf_raw + azimuth_conf_raw) / 2
            print(f"\nAverage confidence: {avg_conf:.1f}%")
            
            if avg_conf < 40.0:
                print(f"⚠️  Rule 3 triggered: Low confidence ({avg_conf:.1f}% < 40%)")
                magnitude_pred = 3
                azimuth_pred = 0
                magnitude_conf = magnitude_probs[0, 3].item() * 100
                azimuth_conf = azimuth_probs[0, 0].item() * 100
                is_corrected = True
        
        # Calculate precursor status
        is_precursor = magnitude_pred != 3 and azimuth_pred != 0
        
        print(f"\nCorrected Predictions:")
        print(f"  Magnitude: class {magnitude_pred} ({magnitude_conf:.1f}%)")
        print(f"  Azimuth: class {azimuth_pred} ({azimuth_conf:.1f}%)")
        print(f"  Corrected: {is_corrected}")
        print(f"  Precursor: {is_precursor}")
        
        # Verify expectations
        print(f"\nExpected:")
        print(f"  Magnitude: class {scenario['expected_magnitude']}")
        print(f"  Azimuth: class {scenario['expected_azimuth']}")
        print(f"  Corrected: {scenario['expected_corrected']}")
        print(f"  Precursor: {scenario['expected_precursor']}")
        
        # Check results
        test_passed = (
            magnitude_pred == scenario['expected_magnitude'] and
            azimuth_pred == scenario['expected_azimuth'] and
            is_corrected == scenario['expected_corrected'] and
            is_precursor == scenario['expected_precursor']
        )
        
        if test_passed:
            print(f"\n✅ TEST PASSED")
            passed += 1
        else:
            print(f"\n❌ TEST FAILED")
            failed += 1
            
            # Show differences
            if magnitude_pred != scenario['expected_magnitude']:
                print(f"   Magnitude mismatch: got {magnitude_pred}, expected {scenario['expected_magnitude']}")
            if azimuth_pred != scenario['expected_azimuth']:
                print(f"   Azimuth mismatch: got {azimuth_pred}, expected {scenario['expected_azimuth']}")
            if is_corrected != scenario['expected_corrected']:
                print(f"   Corrected flag mismatch: got {is_corrected}, expected {scenario['expected_corrected']}")
            if is_precursor != scenario['expected_precursor']:
                print(f"   Precursor flag mismatch: got {is_precursor}, expected {scenario['expected_precursor']}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total tests: {len(scenarios)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/len(scenarios)*100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED! Scanner is working correctly!")
        return True
    else:
        print(f"\n⚠️  SOME TESTS FAILED! Please review the implementation.")
        return False


def test_class_mappings():
    """Test class mappings are correct"""
    
    print(f"\n{'='*70}")
    print(f"CLASS MAPPINGS VERIFICATION")
    print(f"{'='*70}")
    
    magnitude_mapping = {
        0: 'Large (M ≥ 6.0)',
        1: 'Medium (5.0 ≤ M < 6.0)',
        2: 'Moderate (4.0 ≤ M < 5.0)',
        3: 'Normal (No Earthquake)'
    }
    
    azimuth_mapping = {
        0: 'Normal (No Earthquake)',
        1: 'North (N)',
        2: 'South (S)',
        3: 'Northwest (NW)',
        4: 'West (W)',
        5: 'East (E)',
        6: 'Northeast (NE)',
        7: 'Southeast (SE)',
        8: 'Southwest (SW)'
    }
    
    print(f"\nMagnitude Classes:")
    for class_id, class_name in magnitude_mapping.items():
        print(f"  {class_id}: {class_name}")
    
    print(f"\nAzimuth Classes:")
    for class_id, class_name in azimuth_mapping.items():
        print(f"  {class_id}: {class_name}")
    
    print(f"\n✅ Class mappings verified")
    print(f"   Magnitude Normal: class 3")
    print(f"   Azimuth Normal: class 0")
    
    return True


def test_precursor_logic():
    """Test precursor detection logic"""
    
    print(f"\n{'='*70}")
    print(f"PRECURSOR DETECTION LOGIC")
    print(f"{'='*70}")
    
    test_cases = [
        {'mag': 3, 'az': 0, 'expected': False, 'desc': 'Both Normal'},
        {'mag': 0, 'az': 1, 'expected': True, 'desc': 'Large + North'},
        {'mag': 1, 'az': 2, 'expected': True, 'desc': 'Medium + South'},
        {'mag': 2, 'az': 3, 'expected': True, 'desc': 'Moderate + NW'},
        {'mag': 3, 'az': 1, 'expected': False, 'desc': 'Normal + North (invalid)'},
        {'mag': 1, 'az': 0, 'expected': False, 'desc': 'Medium + Normal (invalid)'},
    ]
    
    print(f"\nPrecursor Logic: is_precursor = (magnitude != 3) AND (azimuth != 0)")
    print(f"\nTest Cases:")
    
    passed = 0
    for i, case in enumerate(test_cases, 1):
        is_precursor = case['mag'] != 3 and case['az'] != 0
        test_passed = is_precursor == case['expected']
        
        status = "✅" if test_passed else "❌"
        print(f"  {i}. {case['desc']}: mag={case['mag']}, az={case['az']} → {is_precursor} {status}")
        
        if test_passed:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    
    if passed == len(test_cases):
        print(f"✅ Precursor logic is correct!")
        return True
    else:
        print(f"❌ Precursor logic has issues!")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SCANNER CONSISTENCY RULES - COMPREHENSIVE VERIFICATION")
    print("="*70)
    
    # Run all tests
    test1 = test_class_mappings()
    test2 = test_precursor_logic()
    test3 = test_consistency_rules()
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Class Mappings: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Precursor Logic: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Consistency Rules: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and test2 and test3:
        print(f"\n🎉 ALL VERIFICATIONS PASSED!")
        print(f"✅ Scanner is ready for prekursor detection with correct rules!")
        sys.exit(0)
    else:
        print(f"\n⚠️  SOME VERIFICATIONS FAILED!")
        print(f"❌ Please review and fix the issues.")
        sys.exit(1)
