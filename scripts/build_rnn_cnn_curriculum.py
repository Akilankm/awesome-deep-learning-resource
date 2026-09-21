from pathlib import Path
import json, hashlib, textwrap
import numpy as np
import pandas as pd
import nbformat as nbf
from sklearn.datasets import load_digits

ROOT=Path.cwd()
def md(s): return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(textwrap.dedent(s).strip())
def write_nb(path,cells):
    path.parent.mkdir(parents=True,exist_ok=True)
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}})
    nbf.write(nb,path)

for track in ["RNN","CNN"]:
    for d in ["data","notebooks","projects","scripts","visuals"]:
        (ROOT/track/d).mkdir(parents=True,exist_ok=True)

airline=[112,118,132,129,121,135,148,148,136,119,104,118,115,126,141,135,125,149,170,170,158,133,114,140,145,150,178,163,172,178,199,199,184,162,146,166,171,180,193,181,183,218,230,242,209,191,172,194,196,196,236,235,229,243,264,272,237,211,180,201,204,188,235,227,234,264,302,293,259,229,203,229,242,233,267,269,270,315,364,347,312,274,237,278,284,277,317,313,318,374,413,405,355,306,271,306,315,301,356,348,355,422,465,467,404,347,305,336,340,318,362,348,363,435,491,505,404,359,310,337,360,342,406,396,420,472,548,559,463,407,362,405,417,391,419,461,472,535,622,606,508,461,390,432]
dates=pd.date_range("1949-01-01",periods=len(airline),freq="MS")
pd.DataFrame({"month":dates.strftime("%Y-%m-%d"),"passengers":airline}).to_csv(ROOT/"RNN/data/airline_passengers.csv",index=False)

digits=load_digits()
dfd=pd.DataFrame(digits.data,columns=[f"pixel_{i}" for i in range(64)])
dfd.insert(0,"target",digits.target); dfd.insert(0,"sample_id",np.arange(len(dfd)))
dfd.to_csv(ROOT/"CNN/data/digits.csv",index=False)

COMMON="""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
SEED=42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
ROOT=Path.cwd()
print("TensorFlow version:",tf.__version__)
print("Working directory:",ROOT)
"""

rnn={
"00_rnn_learning_map.ipynb":[md("""# RNN Learning Map
Order and history carry information. The progression is independent rows -> recurrent state -> LSTM/GRU controlled memory. Each notebook connects intuition, shapes, TensorFlow, executed output, interpretation, and engineering consequences."""),code(COMMON),code("""sequence=np.array([10.,12.,15.,14.,18.])
print("Sequence:",sequence)
for t,x in enumerate(sequence): print(f"t={t}: x_t={x}, previous observations={t}")"""),md("""Changing sequence order changes meaning. Window length changes available historical context. Unit count changes representation capacity.""")],
"01_sequence_data_and_shapes.ipynb":[md("""# Sequence Data and Tensor Shapes
RNN inputs follow the conceptual contract (batch, time, features)."""),code(COMMON),code("""x=np.arange(2*4*3).reshape(2,4,3)
print("shape:",x.shape); print("batch 0:",x[0]); print("one timestep:",x[:,0,:].shape)"""),md("""Batch means independent sequences, time means ordered steps, and features means measurements at one step.""")],
"02_windowing_splits_and_leakage.ipynb":[md("""# Windowing, Chronological Splits, and Leakage
Forecasting converts one long series into history-to-future examples. The future test region must remain untouched during development."""),code(COMMON),code("""df=pd.read_csv(ROOT/"RNN/data/airline_passengers.csv"); y=df.passengers.to_numpy(float); w=12
X=np.stack([y[i:i+w] for i in range(len(y)-w)]); Y=np.array([y[i+w] for i in range(len(y)-w)])
print("rows:",len(y),"windows:",X.shape,"targets:",Y.shape); print("first history:",X[0]); print("first target:",Y[0])
a=int(len(X)*.70); b=int(len(X)*.85); print("chronological split sizes:",a,b-a,len(X)-b)"""),md("""Random splitting can leak future regime information. Forecasting validation must respect chronology and the production information boundary.""")],
"03_simple_rnn_hidden_state.ipynb":[md("""# Vanilla RNN and Hidden State
The hidden state is a learned running summary: h_t = tanh(W_x x_t + W_h h_(t-1) + b)."""),code(COMMON),code("""x=np.array([[[1.],[2.],[3.],[4.]]],dtype="float32")
layer=tf.keras.layers.SimpleRNN(3,return_sequences=True,return_state=True); seq,state=layer(x)
print("input:",x.shape,"all states:",seq.shape,"final:",state.shape); print("trajectory:",np.round(seq.numpy(),3))"""),md("""return_sequences exposes every state; turning it off returns only the final representation.""")],
"04_bptt_gradient_flow.ipynb":[md("""# Backpropagation Through Time and Gradient Flow
Repeated transformations create long gradient paths. Multipliers below one shrink; above one grow."""),code(COMMON),code("""steps=np.arange(1,21)
for f in [0.5,0.8,1.0,1.2]: print(f"factor={f}, after20={f**20:.6f}")
plt.figure(figsize=(7,4))
for f in [0.5,0.8,1.0,1.2]: plt.plot(steps,f**steps,label=str(f))
plt.yscale("log"); plt.legend(); plt.xlabel("backward steps"); plt.ylabel("relative gradient"); plt.title("Vanishing / exploding intuition"); plt.show()"""),md("""LSTM/GRU create controlled information paths that make long-dependency learning easier.""")],
"05_lstm_gru_gates.ipynb":[md("""# LSTM and GRU Controlled Memory
LSTM separates memory using a cell state and gates. GRU uses a simpler gated state transition."""),code(COMMON),code("""x=np.random.normal(size=(4,10,2)).astype("float32")
for Layer in [tf.keras.layers.SimpleRNN,tf.keras.layers.LSTM,tf.keras.layers.GRU]:
    layer=Layer(8); y=layer(x); print(Layer.__name__,x.shape,"->",y.shape,"params=",layer.count_params())"""),md("""The external tensor contract can stay the same while the internal memory mechanism changes, enabling controlled comparison.""")],
"06_sequence_architectures.ipynb":[md("""# Sequence Architecture Design
Tasks determine whether we need one output, all-timestep outputs, stacked recurrence, or bidirectional context."""),code(COMMON),code("""inputs=tf.keras.Input(shape=(12,1)); x=tf.keras.layers.LSTM(16,return_sequences=True)(inputs); x=tf.keras.layers.LSTM(8)(x); out=tf.keras.layers.Dense(1)(x); model=tf.keras.Model(inputs,out); model.summary()"""),md("""Bidirectional recurrence is normally invalid for forecasting if it exposes future observations unavailable at inference time.""")],
"07_rnn_training_diagnostics.ipynb":[md("""# RNN Training Diagnostics
Always establish a credible baseline before interpreting neural-network metrics."""),code(COMMON),code("""df=pd.read_csv(ROOT/"RNN/data/airline_passengers.csv"); y=df.passengers.to_numpy(float)
naive=y[:-1]; actual=y[1:]; mae=mean_absolute_error(actual,naive); rmse=mean_squared_error(actual,naive)**0.5
print(f"Naive MAE={mae:.2f}, RMSE={rmse:.2f}"); res=actual-naive; plt.figure(figsize=(8,3)); plt.plot(res); plt.axhline(0,ls="--"); plt.title("Naive residuals"); plt.show()"""),md("""A neural model is useful only relative to the same split and metric contract.""")]
}
for n,c in rnn.items(): write_nb(ROOT/"RNN/notebooks"/n,c)

write_nb(ROOT/"RNN/projects/00_end_to_end_airline_forecasting.ipynb",[
md("""# End-to-End TensorFlow RNN Project: Airline Passenger Forecasting
Lifecycle: raw data -> validation -> EDA -> chronological split -> train-only scaling -> windows -> naive baseline -> SimpleRNN/LSTM/GRU -> validation selection -> untouched test -> residual analysis -> serialization -> reload inference -> monitoring."""),
code(COMMON),
code("""DATA=ROOT/"RNN/data/airline_passengers.csv"; df=pd.read_csv(DATA,parse_dates=["month"])
assert list(df.columns)==["month","passengers"] and df.month.is_monotonic_increasing and not df.isna().any().any()
print(df.head()); print("rows:",len(df),"range:",df.month.min().date(),"to",df.month.max().date()); print(df.passengers.describe().round(2))
plt.figure(figsize=(10,3)); plt.plot(df.month,df.passengers); plt.title("Monthly passengers"); plt.show()"""),
md("""Split policy: last 20 percent is untouched test; preceding 15 percent is validation. Scaling is fitted only on training observations."""),
code("""y=df.passengers.to_numpy(dtype="float32").reshape(-1,1); n=len(y); train_cut=int(n*.65); val_cut=int(n*.80)
scaler=MinMaxScaler().fit(y[:train_cut]); ys=scaler.transform(y).reshape(-1); WINDOW=12
def windows(arr,w):
    X=np.stack([arr[i:i+w] for i in range(len(arr)-w)]); Y=np.array([arr[i+w] for i in range(len(arr)-w)])
    return X[...,None].astype("float32"),Y.astype("float32")
X,Y=windows(ys,WINDOW); idx=np.arange(len(X))+WINDOW
tr=idx<train_cut; va=(idx>=train_cut)&(idx<val_cut); te=idx>=val_cut
Xtr,Ytr,Xv,Yv,Xte,Yte=X[tr],Y[tr],X[va],Y[va],X[te],Y[te]
print("train/val/test:",Xtr.shape,Xv.shape,Xte.shape)"""),
code("""actual=scaler.inverse_transform(Yte.reshape(-1,1)).ravel(); naive=scaler.inverse_transform(Xte[:,-1,0].reshape(-1,1)).ravel(); print("Naive test MAE:",round(mean_absolute_error(actual,naive),2))"""),
code("""def make_model(kind):
    Cell={"SimpleRNN":tf.keras.layers.SimpleRNN,"LSTM":tf.keras.layers.LSTM,"GRU":tf.keras.layers.GRU}[kind]
    m=tf.keras.Sequential([tf.keras.layers.Input((WINDOW,1)),Cell(24),tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss="mse",metrics=["mae"]); return m
scores={}; models={}
for kind in ["SimpleRNN","LSTM","GRU"]:
    m=make_model(kind); h=m.fit(Xtr,Ytr,validation_data=(Xv,Yv),epochs=35,batch_size=16,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)])
    scores[kind]=float(m.evaluate(Xv,Yv,verbose=0)[0]); models[kind]=m; print(kind,"val loss:",round(scores[kind],5),"epochs:",len(h.history["loss"]))
best=min(scores,key=scores.get); model=models[best]; print("Selected using validation only:",best)"""),
code("""pred_scaled=model.predict(Xte,verbose=0).reshape(-1,1); pred=scaler.inverse_transform(pred_scaled).ravel()
mae=mean_absolute_error(actual,pred); rmse=mean_squared_error(actual,pred)**0.5
print(f"FINAL TEST {best}: MAE={mae:.2f}, RMSE={rmse:.2f}")
plt.figure(figsize=(9,3)); plt.plot(actual,label="actual",marker="o"); plt.plot(pred,label="forecast",marker="o"); plt.legend(); plt.title("Untouched test"); plt.show()
err=actual-pred; print("largest absolute errors:",np.sort(np.abs(err))[-5:][::-1].round(2))"""),
code("""art=ROOT/"RNN/projects/artifacts"; rep=ROOT/"RNN/projects/reports"; art.mkdir(exist_ok=True); rep.mkdir(exist_ok=True)
model.save(art/"best_rnn_forecaster.keras")
(rep/"metrics.json").write_text(json.dumps({"selected_model":best,"test_mae":float(mae),"test_rmse":float(rmse),"window":WINDOW},indent=2))
pd.DataFrame({"actual":actual,"prediction":pred,"error":err}).to_csv(rep/"predictions.csv",index=False)
reloaded=tf.keras.models.load_model(art/"best_rnn_forecaster.keras"); check=reloaded.predict(Xte[:1],verbose=0)
print("saved/reloaded model; next prediction:",np.round(scaler.inverse_transform(check).ravel(),1))"""),
md("""Production interpretation: preserve the 12-step input contract and training-fitted scaler; monitor input drift, missing observations, recent residual metrics, and retrain on sustained degradation.""")
])

cnn={
"00_cnn_learning_map.ipynb":[md("""# CNN Learning Map
Pixels -> local filters -> feature maps -> larger receptive fields -> representation -> classifier. Each notebook combines tensor geometry, TensorFlow/Keras, executed outputs, and engineering interpretation."""),code(COMMON),code("""df=pd.read_csv(ROOT/"CNN/data/digits.csv"); print("rows:",len(df),"classes:",sorted(df.target.unique())); img=df.filter(like="pixel_").iloc[0].to_numpy().reshape(8,8); plt.imshow(img,cmap="gray"); plt.title(f"label={df.target.iloc[0]}"); plt.axis("off"); plt.show()"""),md("""Changing kernel weights changes detected patterns; filter count changes detector capacity; pooling and stride change spatial resolution.""")],
"01_images_channels_shapes.ipynb":[md("""# Images, Channels, and Tensor Shapes
TensorFlow Conv2D uses (batch, height, width, channels)."""),code(COMMON),code("""x=np.zeros((32,8,8,1),dtype="float32"); print("batch:",x.shape); print("one image:",x[0].shape); print("one row:",x[0,0].shape)"""),md("""A kernel spans all input channels while sliding over height and width.""")],
"02_convolution_from_first_principles.ipynb":[md("""# Convolution from First Principles
A feature-map value is a local weighted dot product. Libraries normally implement cross-correlation while retaining the conventional name convolution."""),code(COMMON),code("""image=np.array([[1,2,0,0],[0,1,3,1],[2,1,0,2],[1,0,2,1]],float); kernel=np.array([[1,0],[-1,1]],float); out=np.empty((3,3))
for i in range(3):
    for j in range(3): out[i,j]=(image[i:i+2,j:j+2]*kernel).sum()
print("image:",image); print("kernel:",kernel); print("feature map:",out)"""),md("""Training learns kernel weights from the task loss.""")],
"03_stride_padding_receptive_field.ipynb":[md("""# Stride, Padding, and Receptive Field
Output size in one dimension is floor((N + 2P - K) / S) + 1."""),code(COMMON),code("""def out_size(n,k,s=1,p=0): return (n+2*p-k)//s+1
for cfg in [(8,3,1,0),(8,3,1,1),(8,3,2,1)]: print(cfg,"->",out_size(*cfg))
x=np.zeros((1,8,8,1),dtype="float32")
for padding,strides in [("valid",1),("same",1),("same",2)]:
    y=tf.keras.layers.Conv2D(4,3,padding=padding,strides=strides)(x); print(padding,strides,"=>",y.shape)"""),md("""Stride changes resolution and compute. Padding controls boundary geometry. Stacked layers expand receptive field.""")],
"04_filters_feature_maps_pooling.ipynb":[md("""# Filters, Feature Maps, and Pooling
One filter produces one output channel; multiple filters learn different local evidence."""),code(COMMON),code("""x=np.random.rand(2,8,8,1).astype("float32"); conv=tf.keras.layers.Conv2D(6,3,activation="relu"); pool=tf.keras.layers.MaxPooling2D(); a=conv(x); b=pool(a); print("input:",x.shape,"conv:",a.shape,"pooled:",b.shape,"params:",conv.count_params())"""),md("""Parameters do not grow with image width/height because convolution shares weights spatially.""")],
"05_building_cnn_tensorflow.ipynb":[md("""# Building a CNN in TensorFlow/Keras
The geometric pieces now become one executable model."""),code(COMMON),code("""model=tf.keras.Sequential([tf.keras.layers.Input((8,8,1)),tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),tf.keras.layers.Flatten(),tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")]); model.summary(); print("output:",model(np.zeros((5,8,8,1),dtype="float32")).shape)"""),md("""Students should predict layer output shapes before executing them.""")],
"06_regularization_augmentation.ipynb":[md("""# Regularization and Augmentation
Augmentation encodes the assumption that a transformed image keeps its class."""),code(COMMON),code("""aug=tf.keras.Sequential([tf.keras.layers.RandomTranslation(.05,.05),tf.keras.layers.RandomZoom(.05)]); x=np.random.rand(4,8,8,1).astype("float32"); y=aug(x,training=True); print("shape:",x.shape,"->",y.shape); print("mean absolute change:",float(np.mean(np.abs(y.numpy()-x))))"""),md("""For digits, aggressive flips or rotations may change identity, so augmentation must be justified.""")],
"07_interpretability_error_analysis.ipynb":[md("""# Interpretability and Error Analysis
Accuracy hides where the model fails. Use per-class metrics, confusion matrices, confidence, wrong examples, and intermediate activations."""),code(COMMON),code("""cm=np.array([[45,2,0],[3,40,4],[0,5,42]]); print("confusion matrix:",cm); print("recall:",np.round(np.diag(cm)/cm.sum(axis=1),3)); plt.imshow(cm); plt.title("Reading a confusion matrix"); plt.xlabel("predicted"); plt.ylabel("actual"); plt.colorbar(); plt.show()"""),md("""The project computes these diagnostics on the untouched test set and inspects learned feature maps.""")]
}
for n,c in cnn.items(): write_nb(ROOT/"CNN/notebooks"/n,c)

write_nb(ROOT/"CNN/projects/00_end_to_end_digits_cnn.ipynb",[
md("""# End-to-End TensorFlow CNN Project: Handwritten Digit Classification
Lifecycle: raw CSV -> validation -> EDA -> stratified split -> normalization -> dense baseline -> CNN -> validation selection -> untouched test -> confusion/error analysis -> feature maps -> serialization -> reload inference -> monitoring."""),
code(COMMON+"""
from sklearn.model_selection import train_test_split
"""),
code("""DATA=ROOT/"CNN/data/digits.csv"; df=pd.read_csv(DATA); pixels=[c for c in df if c.startswith("pixel_")]
assert len(pixels)==64 and df[pixels].isna().sum().sum()==0
X=df[pixels].to_numpy(dtype="float32").reshape(-1,8,8,1); y=df.target.to_numpy(dtype="int64")
print("X/y:",X.shape,y.shape,"range:",X.min(),X.max()); print(df.target.value_counts().sort_index())
fig,axs=plt.subplots(2,5,figsize=(8,4))
for ax,i in zip(axs.ravel(),range(10)):
    j=np.where(y==i)[0][0]; ax.imshow(X[j,:,:,0],cmap="gray"); ax.set_title(str(i)); ax.axis("off")
plt.tight_layout(); plt.show()"""),
md("""Split contract: stratified train/validation/test. Test stays untouched until selection. Pixel scaling uses the known source range 0 to 16."""),
code("""X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=SEED,stratify=y); X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=.1875,random_state=SEED,stratify=y_train); X_train/=16.; X_val/=16.; X_test/=16.; print("train/val/test:",len(X_train),len(X_val),len(X_test))"""),
code("""def baseline():
    m=tf.keras.Sequential([tf.keras.layers.Input((8,8,1)),tf.keras.layers.Flatten(),tf.keras.layers.Dense(10,activation="softmax")]); m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return m
def cnn_model():
    m=tf.keras.Sequential([tf.keras.layers.Input((8,8,1)),tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),tf.keras.layers.Flatten(),tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")]); m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return m
models={"Dense baseline":baseline(),"CNN":cnn_model()}; scores={}
for name,m in models.items():
    h=m.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=30,batch_size=32,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
    scores[name]=float(m.evaluate(X_val,y_val,verbose=0)[1]); print(name,"validation accuracy:",round(scores[name],4),"epochs:",len(h.history["loss"]))
best=max(scores,key=scores.get); model=models[best]; print("Selected using validation only:",best)"""),
code("""probs=model.predict(X_test,verbose=0); pred=probs.argmax(1); acc=accuracy_score(y_test,pred); cm=confusion_matrix(y_test,pred)
print("FINAL TEST ACCURACY:",round(acc,4)); print(classification_report(y_test,pred,digits=3))
plt.figure(figsize=(6,5)); plt.imshow(cm); plt.title("Untouched test confusion matrix"); plt.xlabel("predicted"); plt.ylabel("actual"); plt.colorbar(); plt.show()
wrong=np.where(pred!=y_test)[0]; print("misclassified:",len(wrong),"of",len(y_test)); print("first errors:",[(int(y_test[i]),int(pred[i]),round(float(probs[i].max()),3)) for i in wrong[:10]])"""),
code("""conv=next((l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)),None)
if conv is not None:
    probe=tf.keras.Model(model.inputs[0],conv.output); fmap=probe.predict(X_test[:1],verbose=0)[0]; print("feature maps:",fmap.shape)
    fig,axs=plt.subplots(2,4,figsize=(8,4))
    for k,ax in enumerate(axs.ravel()): ax.imshow(fmap[:,:,k],cmap="viridis"); ax.axis("off"); ax.set_title(f"filter {k}")
    plt.tight_layout(); plt.show()
else: print("Dense baseline selected; no feature maps.")"""),
code("""art=ROOT/"CNN/projects/artifacts"; rep=ROOT/"CNN/projects/reports"; art.mkdir(exist_ok=True); rep.mkdir(exist_ok=True)
model.save(art/"best_digit_classifier.keras"); (rep/"metrics.json").write_text(json.dumps({"selected_model":best,"test_accuracy":float(acc)},indent=2)); pd.DataFrame({"actual":y_test,"prediction":pred,"confidence":probs.max(1)}).to_csv(rep/"predictions.csv",index=False)
reloaded=tf.keras.models.load_model(art/"best_digit_classifier.keras"); check=reloaded.predict(X_test[:1],verbose=0); print("saved/reloaded; prediction:",int(check.argmax(1)[0]),"actual:",int(y_test[0]),"confidence:",round(float(check.max()),3))"""),
md("""Production interpretation: monitor class mix, pixel distribution, confidence drift, latency, and delayed-label accuracy. Retraining should respond to sustained degradation.""")
])

env="""channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - plotly
  - ipywidgets
  - jupyterlab
  - notebook
  - ipykernel
  - nbformat
  - nbclient
  - nbconvert
  - pyarrow
  - tqdm
  - pip:
      - tensorflow==2.20.*
"""
(ROOT/"RNN/environment.yml").write_text("name: awesome-rnn\\n\\n"+env); (ROOT/"CNN/environment.yml").write_text("name: awesome-cnn\\n\\n"+env)
(ROOT/"RNN/README.md").write_text("# RNN - TensorFlow/Keras\\nExecuted student-first curriculum from sequence tensors through LSTM/GRU and an end-to-end forecasting project.\\n\\nRun: conda env create -f RNN/environment.yml; conda activate awesome-rnn; jupyter lab\\n")
(ROOT/"CNN/README.md").write_text("# CNN - TensorFlow/Keras\\nExecuted student-first curriculum from convolution fundamentals through an end-to-end CNN image project.\\n\\nRun: conda env create -f CNN/environment.yml; conda activate awesome-cnn; jupyter lab\\n")
(ROOT/"RNN/data/DATASET_CARD.md").write_text("# Airline Passengers Dataset Card\\n144 monthly observations, 1949-01 to 1960-12. Target: passengers. No missing values. Teaching focus: trend, seasonality, temporal dependence, leakage-safe splits.\\n")
(ROOT/"RNN/data/README.md").write_text("Committed locally for offline reproducibility.\\n")
(ROOT/"CNN/data/DATASET_CARD.md").write_text("# Handwritten Digits Dataset Card\\nscikit-learn digits dataset: 1,797 samples, 8x8 grayscale, classes 0-9, pixel range 0-16. Teaching focus: spatial tensors, convolution, baseline comparison, feature maps, error analysis.\\n")
(ROOT/"CNN/data/README.md").write_text("Committed locally for offline reproducibility.\\n")
for track,fn in [("RNN","airline_passengers.csv"),("CNN","digits.csv")]:
    p=ROOT/track/"data"/fn; (ROOT/track/"data/checksums.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {fn}\\n")

exec_script="""from pathlib import Path
import sys,nbformat
from nbclient import NotebookClient
track=Path(sys.argv[1]).resolve(); root=track.parent
paths=sorted(list((track/"notebooks").glob("*.ipynb"))+list((track/"projects").glob("*.ipynb")))
for p in paths:
    print("Executing",p); nb=nbformat.read(p,as_version=4); NotebookClient(nb,timeout=900,kernel_name="python3",resources={"metadata":{"path":str(root)}}).execute(); nbformat.write(nb,p)
print("Executed",len(paths),"notebooks")
"""
verify="""from pathlib import Path
import sys,nbformat
track=Path(sys.argv[1]); problems=[]
for p in sorted(list((track/"notebooks").glob("*.ipynb"))+list((track/"projects").glob("*.ipynb"))):
    nb=nbformat.read(p,as_version=4); cells=[c for c in nb.cells if c.cell_type=="code"]
    if any(c.execution_count is None for c in cells): problems.append(f"{p}: unexecuted cells")
    if any(o.output_type=="error" for c in cells for o in c.get("outputs",[])): problems.append(f"{p}: error output")
    print("OK",p)
if problems: print("\\n".join(problems)); raise SystemExit(1)
print("All notebooks are executed and error-free.")
"""
for track in ["RNN","CNN"]:
    (ROOT/track/"scripts/execute_notebooks.py").write_text(exec_script); (ROOT/track/"scripts/verify_notebooks.py").write_text(verify); (ROOT/track/"projects/README.md").write_text(f"# {track} Projects\\nThe end-to-end notebook runs from committed data to saved model, reports, reload inference, and production considerations.\\n")

(ROOT/"README.md").write_text("# Awesome Deep Learning Resource\\n\\nVisual, executable deep-learning curriculum. Existing ANN and NLP material is joined by TensorFlow-only RNN and CNN tracks. RNN/CNN dependency management is Conda-first. Committed notebooks are executed and include rendered outputs. Each flagship project runs from committed data through validation, model selection, untouched test evaluation, serialization, reload inference, and monitoring/retraining guidance.\\n")

wf="""name: RNN CNN notebook validation
on:
  push:
    branches: [master]
    paths:
      - 'RNN/**'
      - 'CNN/**'
      - '.github/workflows/rnn_cnn_notebooks.yml'
  pull_request:
    paths:
      - 'RNN/**'
      - 'CNN/**'
jobs:
  rnn:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: conda-incubator/setup-miniconda@v3
        with:
          environment-file: RNN/environment.yml
          activate-environment: awesome-rnn
          auto-activate-base: false
      - shell: bash -el {0}
        run: python -c "import tensorflow as tf; print(tf.__version__)"
      - shell: bash -el {0}
        run: python RNN/scripts/execute_notebooks.py RNN
      - shell: bash -el {0}
        run: python RNN/scripts/verify_notebooks.py RNN
  cnn:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: conda-incubator/setup-miniconda@v3
        with:
          environment-file: CNN/environment.yml
          activate-environment: awesome-cnn
          auto-activate-base: false
      - shell: bash -el {0}
        run: python -c "import tensorflow as tf; print(tf.__version__)"
      - shell: bash -el {0}
        run: python CNN/scripts/execute_notebooks.py CNN
      - shell: bash -el {0}
        run: python CNN/scripts/verify_notebooks.py CNN
"""
(ROOT/".github/workflows").mkdir(parents=True,exist_ok=True); (ROOT/".github/workflows/rnn_cnn_notebooks.yml").write_text(wf)
print("Generated complete RNN/CNN curriculum tree.")
