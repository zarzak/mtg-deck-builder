#!/usr/bin/env python3
"""
Debug script to diagnose card database issues.
Run with: python debug_db.py cards.csv
"""

import sys
import csv
from pathlib import Path
from io import StringIO


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_db.py <path-to-cards.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print(f"Analyzing: {csv_path}")
    print("=" * 60)
    
    # Check for BOM and raw bytes
    with open(csv_path, 'rb') as f:
        first_bytes = f.read(10)
        print(f"\nFirst 10 bytes (raw): {first_bytes}")
        
        if first_bytes.startswith(b'\xef\xbb\xbf'):
            print("⚠️  UTF-8 BOM detected (will be handled automatically)")
        elif first_bytes.startswith(b'\xff\xfe') or first_bytes.startswith(b'\xfe\xff'):
            print("⚠️  UTF-16 BOM detected! File may need conversion.")
    
    # Read and filter out comment lines
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        all_lines = f.readlines()
    
    comment_lines = [l for l in all_lines if l.strip().startswith('#')]
    data_lines = [l for l in all_lines if l.strip() and not l.strip().startswith('#')]
    
    print(f"\nFile structure:")
    print(f"  Total lines: {len(all_lines)}")
    print(f"  Comment lines (starting with #): {len(comment_lines)}")
    print(f"  Data lines: {len(data_lines)}")
    
    if not data_lines:
        print("\n❌ No data lines found!")
        return
    
    # Analyze header line (FIRST NON-COMMENT LINE)
    header_line = data_lines[0].strip()
    print(f"\nHeader line (first non-comment line):")
    print(f"  '{header_line[:100]}...'")
    
    # Check delimiter
    if '|' in header_line:
        print("\n✓ Pipe delimiter detected")
        delimiter = '|'
    elif '\t' in header_line:
        print("\n✓ Tab delimiter detected")
        delimiter = '\t'
    elif ',' in header_line:
        print("\n⚠ Comma delimiter detected")
        delimiter = ','
    else:
        print("\n❌ Unknown delimiter")
        delimiter = '|'
    
    # Parse columns from header
    cols = header_line.split(delimiter)
    print(f"  Columns ({len(cols)}): {cols[:8]}...")
    
    if 'name' in cols:
        print("  ✓ 'name' column found")
    else:
        print(f"  ❌ 'name' column NOT found!")
    
    # Parse as CSV with proper quoting (using only data lines)
    csv_content = ''.join(data_lines)
    reader = csv.DictReader(
        StringIO(csv_content),
        delimiter=delimiter,
        quotechar='"',
        doublequote=True
    )
    
    rows = list(reader)
    print(f"\n✓ Parsed {len(rows)} cards")
    
    # Show first few cards
    print("\n" + "=" * 60)
    print("First 5 cards:")
    print("=" * 60)
    for i, row in enumerate(rows[:5]):
        name = row.get('name', 'NO_NAME')
        mana = row.get('manaCost', '')
        print(f"  {i+1}. {name} {mana}")
    
    # Search for test commanders
    search_terms = ["lathiel", "jasmine", "karlov", "boreal"]
    
    print("\n" + "=" * 60)
    print("Searching for test commanders...")
    print("=" * 60)
    
    for term in search_terms:
        matches = [
            row.get('name', 'NO_NAME') 
            for row in rows 
            if term.lower() in row.get('name', '').lower()
        ]
        if matches:
            print(f"\n'{term}' matches:")
            for m in matches[:5]:
                print(f"  → '{m}'")
        else:
            print(f"\n'{term}': No matches found")
    
    # Check legalities
    print("\n" + "=" * 60)
    print("Checking legalities...")
    print("=" * 60)
    
    sample = rows[0] if rows else {}
    legalities = sample.get('legalities', 'NOT_FOUND')
    print(f"\nSample legalities: '{legalities[:80] if legalities else 'EMPTY'}'")
    
    commander_legal = [r for r in rows if 'commander' in r.get('legalities', '').lower()]
    print(f"Commander-legal cards: {len(commander_legal)} / {len(rows)}")
    
    # Test exact lookup
    print("\n" + "=" * 60)
    print("Testing exact name lookup:")
    print("=" * 60)
    
    test_name = "Lathiel, the Bounteous Dawn"
    print(f"\nLooking for: '{test_name}'")
    
    by_name = {row.get('name', '').lower().strip(): row for row in rows}
    lookup_key = test_name.lower().strip()
    
    if lookup_key in by_name:
        found = by_name[lookup_key]
        print(f"✓ FOUND!")
        print(f"  Name: {found.get('name')}")
        print(f"  Mana: {found.get('manaCost')}")
        print(f"  Type: {found.get('type')}")
    else:
        print(f"❌ NOT FOUND")
        similar = [k for k in by_name.keys() if 'lathiel' in k]
        if similar:
            print(f"  Similar keys: {similar}")
        # Show some actual keys to debug
        print(f"\n  First 5 keys in lookup dict:")
        for k in list(by_name.keys())[:5]:
            print(f"    '{k}'")
    
    # Interactive search
    print("\n" + "=" * 60)
    print("Interactive search (type card name, or 'quit' to exit):")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nSearch: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            
            matches = [
                row.get('name', 'NO_NAME')
                for row in rows
                if query.lower() in row.get('name', '').lower()
            ]
            
            if matches:
                print(f"Found {len(matches)} matches:")
                for m in matches[:10]:
                    print(f"  → '{m}'")
                if len(matches) > 10:
                    print(f"  ... and {len(matches) - 10} more")
            else:
                print("No matches found")
                
        except EOFError:
            break


if __name__ == "__main__":
    main()
