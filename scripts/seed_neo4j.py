from src.graph import EXAMPLE_CYPHER_QUERIES, cypher_query, seed_graph_from_mongo

if __name__ == '__main__':
    print('Neo4j seed:', seed_graph_from_mongo())
    for item in EXAMPLE_CYPHER_QUERIES:
        print('\n---', item['name'], '---')
        print(item['cypher'])
        try:
            rows = cypher_query(item['cypher'], item.get('parameters'))
            print(rows[:5])
        except Exception as e:
            print('Query skipped/error:', e)
