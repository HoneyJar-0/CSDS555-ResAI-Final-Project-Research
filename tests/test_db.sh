python -c "from dataset_pipeline.dataset_creation import pipeline; pipeline()"
python -c "
import duckdb
c=duckdb.connect('data/input/dataset.duckdb',read_only=True)
print('Identities:', c.execute('SELECT COUNT(*) FROM identities').fetchone())
print('Scenarios:', c.execute('SELECT COUNT(*) FROM scenarios').fetchone())
print('Stories:', c.execute('SELECT COUNT(*) FROM stories').fetchone())
"

python -m llm_pipeline.benchmark
python -c "import duckdb
c=duckdb.connect('data/input/dataset.duckdb',read_only=True)
print('Responses:', c.execute('SELECT COUNT(*) FROM responses').fetchone())
print(c.execute('SELECT * FROM responses LIMIT 3').df())
"

python -m evaluation_pipeline.evaluation
python -c "
import duckdb
c = duckdb.connect('data/input/dataset.duckdb',read_only=True)
print('Evaluations:', c.execute('SELECT COUNT(*) FROM evaluations').fetchone())
print(c.execute('SELECT * FROM evaluations LIMIT 3').df())
"

python -c "
import duckdb
c = duckdb.connect('data/input/dataset.duckdb', read_only=True)
print(c.execute('''
    SELECT
        s.id AS story_id,
        si.identity AS system_identity,
        bi.identity AS subject_identity,
        sc.template AS scenario,
        r.model,
        r.response,
        e.positive, e.negative, e.neutral, e.signed_bias, e.is_blocked
    FROM evaluations e
    JOIN responses r   ON r.id  = e.response_id
    JOIN stories s     ON s.id  = r.story_id
    JOIN identities si ON si.id = s.system_identity_id
    JOIN identities bi ON bi.id = s.subject_identity_id
    JOIN scenarios sc  ON sc.id = s.scenario_id
    LIMIT 5
''').df().to_string())
"
