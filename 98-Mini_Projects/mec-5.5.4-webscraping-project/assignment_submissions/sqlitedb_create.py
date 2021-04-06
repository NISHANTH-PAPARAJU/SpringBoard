import sqlite3
import json


def create_tables(c, conn):
    """
    Creates the quotes & quotetags tables in the database
    """
    c.execute("""CREATE TABLE quotes (
                id INTEGER PRIMARY KEY,
                text TEXT,
                author TEXT
                )""")
    c.execute("""CREATE TABLE quotetags (
                q_id INTEGER,
                tag TEXT
                )""")
    conn.commit()


def insert_data(f):
    # insert data into tables from the JSON file
    read_file = open(f, 'r')
    quotes_json = json.load(read_file)
    idx = 1
    for quote in quotes_json:
        text = str.replace(quote["text"], "'", "''")
        author = str.replace(quote["author"], "'", "''")
        tags = quote["tags"]
        # Insert data into quotes table
        c.execute(f"INSERT INTO quotes VALUES ({idx}, '{text}', '{author}')")
        # Insert data in to quotetags table
        for tag in tags:
            tag_formatted = str.replace(tag, "'", "''")
            c.execute(f"INSERT INTO quotetags VALUES ({idx}, '{tag_formatted}')")
        idx += 1

    conn.commit()


conn = sqlite3.connect('results-sqlite3db/quotes.db')
c = conn.cursor()
file = 'results/css-scaper-results.json'
# create_tables(c, conn)
# insert_data(file)

c.execute("""SELECT q.text, q.author, t.tag
            FROM quotes q
            JOIN quotetags t ON q.id = t.q_id
            WHERE t.tag='inspirational' """)
print(c.fetchall())



