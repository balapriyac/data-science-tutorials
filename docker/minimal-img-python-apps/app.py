from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory database for simplicity
inventory = {}

@app.route('/inventory', methods=['POST'])
def add_item():
    item = request.get_json()
    item_id = item.get('id')
    if not item_id:
        return jsonify({"error": "Item ID is required"}), 400
    if item_id in inventory:
        return jsonify({"error": "Item already exists"}), 400
    inventory[item_id] = item
    return jsonify(item), 201

@app.route('/inventory/<item_id>', methods=['GET'])
def get_item(item_id):
    item = inventory.get(item_id)
    if not item:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)

@app.route('/inventory/<item_id>', methods=['PUT'])
def update_item(item_id):
    if item_id not in inventory:
        return jsonify({"error": "Item not found"}), 404
    updated_item = request.get_json()
    inventory[item_id] = updated_item
    return jsonify(updated_item)

@app.route('/inventory/<item_id>', methods=['DELETE'])
def delete_item(item_id):
    if item_id not in inventory:
        return jsonify({"error": "Item not found"}), 404
    del inventory[item_id]
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
