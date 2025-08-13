from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from grid_rl_env import GridEnv, train_agent  # your RL environment

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    grid = data['grid']
    env = GridEnv(grid)
    path, q_table, delivery_log = train_agent(env)

    # Convert q_table keys to strings
    serializable_q_table = {str(k): v for k, v in q_table.items()}

    return jsonify({
        'path': path,
        'q_table': serializable_q_table,
        'delivery_log': delivery_log
    })


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
