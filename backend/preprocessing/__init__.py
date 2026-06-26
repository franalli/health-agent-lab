"""Ingestion adapter (the one normalization firewall): member-bundle JSON -> validated SQLite.

Everything raw enters through here exactly once; nothing downstream re-parses input.
"""
