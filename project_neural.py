if __name__ == "__main__":
    from flask import Flask, app, request, session, render_template, redirect, send_from_directory, make_response
    import network as net_gen
else:
    from philsite import app, request, session, render_template, redirect, send_from_directory, make_response
    import philsite.project_neural.network as net_gen
import os, time
from functools import wraps, update_wrapper
from datetime import datetime

#Run as Standalone
if __name__ == "__main__":
    app_dir = os.path.dirname(os.path.realpath(__file__))       #Current Directory
    path = "/"                                                  #Path prefix for website
    dir_name="/"                                                #Directory for python ilfe
    static_dir = app_dir + "/static"                            #Directory for static files

    bind = "127.0.0.1"                                          #Local IP
    port = 5000                                                 #Port to run on
    app = Flask(__name__, template_folder="")

#Run as part of main website
else:
    app_dir = philsite.app_dir                                 #Directory of __init__ python file
    path = "/neural_net"                                       #Url Path
    dir_name="project_neural/"                                 #Directory of project on website
    static_dir = app_dir + "/philsite/project_neural/static"   #Directory for static files

network_dir = os.path.dirname(os.path.realpath(__file__))  #Directory for network object to use for storing resources
#network_dir = "" if network_dir == "/" else network_dir    #Edge (not really) case

#Initial network generated on startup
network = net_gen.Network([2, 1, 1, 3], threshold=0.5, path=network_dir)
network.export_graph()
training_data = net_gen.TrainingData(network, [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])])

def nocache(view):
    """Wrapper to disable caching on a page (Currently Nonfuctional? To be worked on)"""
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Cache_Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "-1"
        return response

    return update_wrapper(no_cache, view)

@app.route("/neural_static/<path:filename>")
def neural_static(filename):
    """Serves static files"""
    return send_from_directory(static_dir, filename)

@app.route(path)
@nocache
def neural_index():
    """Main page"""
    return render_template(dir_name+"templates/neural_net.html", network=network, training_data=training_data, path=path)


@app.route(path+"new_net", methods=["GET", "POST"])
def generate_new_net():
    """Generates a new network"""
    global network
    if request.method == "POST":
        print(request.form)
        data = request.form
        network = net_gen.Network([int(data["inputs"]),
                                    int(data["outputs"]),
                                    int(data["hidden_layers"]),
                                    int(data["hidden_neurons"])],
                                    threshold=0.5,
                                    path=network_dir)
        network.export_graph()
    return redirect(path)

@app.route(path+"propagate", methods=["GET", "POST"])
def propagate():
    if request.method == "POST":
        data = request.form
        inputs = [(int(x[-1]), int(y)) for x, y in data.items() if x[:5] == "input"]
        inputs = [y for x, y in sorted(inputs, key=lambda tup: tup[0])]
        network.propagate(inputs)
        network.export_graph()
    return redirect(path)

@app.route(path+"darwin_train", methods=["GET", "POST"])
def darwin_train():
    if request.method == "POST":
        global network
        reps = int(request.form["repititions"])
        if reps > 200:
            reps = 200
        trainer = net_gen.RandomTrainer(2)
        network = trainer.train(network, training_data.data, reps)
        network.export_graph(print_values=False)
    return redirect(path)

@app.route(path+"new_data", methods=["GET", "POST"])
def add_data():
    if request.method == "POST":
        data = request.form
        print(data)
        inputs = [(int(x[-1]), int(y)) for x, y in data.items() if x[:5] == "input"]
        inputs = [y for x, y in sorted(inputs, key=lambda tup: tup[0])]
        outputs = [(int(x[-1]), int(y)) for x, y in data.items() if x[:6] == "output"]
        outputs = [y for x, y in sorted(outputs, key=lambda tup: tup[0])]
        training_data.add_data(inputs, outputs)
    return redirect(path)

@app.route(path+"clear_data")
def clear_data():
    training_data.clear()
    return redirect(path)

if __name__ == "__main__":
    app.run(host=bind, port=port) #This line needs to be here apparently
