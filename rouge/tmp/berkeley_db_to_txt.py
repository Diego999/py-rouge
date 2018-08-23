from bsddb3 import db

# Generate key_value from Berkeley DB
db_path = 'WordNet-2.0.exc.db'
output_path = 'wordnet_key_value.txt'

DB = db.DB()
DB.open(db_path)

cursor = DB.cursor()

with open(output_path, 'w', encoding='utf-8') as fp:
    rec = cursor.first()
    while rec:
        fp.write('{}|{}\n'.format(rec[0].decode('utf-8'), rec[1].decode('utf-8')))
        rec = cursor.next()
DB.close()