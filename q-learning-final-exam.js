class RandomGenerator {
  constructor(seed) {
    this.generator = new Math.seedrandom(seed || '.');
  }
  random() {
    return this.generator();
  }
  setSeed(seed) {
    this.generator = new Math.seedrandom(seed);
  }
}

const generator = new RandomGenerator();

class Action {
  constructor(nextState, reward, name) {
    this.nextState = nextState;
    this.reward = reward;
    this.name = name;
  }
}

class State {
  constructor(name) {
    this.name = name;
    this.actions = {};
    this.actionsList = [];
  }

  addAction(nextState, reward, name) {
    let action = new Action(nextState, reward, name);
    this.actions[name] = action;
    this.actionsList.push(action);
    return action;
  }

  randomAction = () => {
    let index = Math.floor(this.actionsList.length * generator.random());
    return this.actionsList[index];
  };
}

class QLearner {
  constructor(gamma, alpha) {
    this.gamma = gamma || 0.8;
    this.alpha = alpha || 0.8;
    this.qValuesTable = {};
    this.states = {};
    this.statesList = [];
    this.currentState = null;
  }

  add(from, to, reward, actionName) {
    if (!this.states[from]) this.addState(from);
    if (!this.states[to]) this.addState(to);
    this.setQValue(from, actionName, reward);
    return this.states[from].addAction(to, reward, actionName);
  }

  addState(name) {
    let state = new State(name);
    this.states[name] = state;
    this.statesList.push(state);
    return state;
  }

  setState(name) {
    this.currentState = this.states[name] || new State(name);
    return this.currentState;
  }

  getState() {
    return this.currentState && this.currentState.name;
  }

  randomState() {
    let index = Math.floor(this.statesList.length * generator.random());
    return this.statesList[index];
  }

  optimalFutureValue(state) {
    let qValues = this.qValuesTable[state];
    let max = 0;
    for (let action in qValues) {
      if (qValues.hasOwnProperty(action)) {
        max = Math.max(max, qValues[action] || 0);
      }
    }
    return max;
  }

  knowsAction(state, action) {
    return (this.qValuesTable[state] || {}).hasOwnProperty(action);
  }

  applyAction(name) {
    let actionObject = this.states[this.currentState.name].actions[name];
    if (actionObject) {
      this.currentState = this.states[actionObject.nextState];
    }
    return actionObject && this.currentState;
  }

  bestAction(state) {
    let qValues = this.qValuesTable[state] || {};
    let bestAction = null;
    for (let action in qValues) {
      if (qValues.hasOwnProperty(action)) {
        if (!bestAction) {
          bestAction = action;
        } else if (
          qValues[action] == qValues[bestAction] &&
          generator.random() > 0.5
        ) {
          bestAction = action;
        } else if (qValues[action] > qValues[bestAction]) {
          bestAction = action;
        }
      }
    }
    return bestAction;
  }

  runOnce() {
    let bestActionName = this.bestAction(this.currentState.name);
    if (!bestActionName) return undefined;
    let action = this.states[this.currentState.name].actions[bestActionName];
    if (action) {
      this.currentState = this.states[action.nextState];
    }
    return action && this.currentState;
  }

  setQValue(stateName, actionName, reward) {
    if (!this.qValuesTable[stateName]) this.qValuesTable[stateName] = {};
    this.qValuesTable[stateName][actionName] = reward;
  }

  getQValue(stateName, actionName) {
    if (!this.qValuesTable[stateName]) return 0.0;
    return this.qValuesTable[stateName][actionName] || 0.0;
  }

  step() {
    if (!this.currentState) this.currentState = this.randomState();
    if (!this.currentState) return null;
    let state = this.currentState;
    let action = state.randomAction();
    if (!action) return null;
    let actionReward = action.reward || 0;
    let maxQValue = this.optimalFutureValue(action.nextState);
    let oldQValue = this.getQValue(state.name, action.name);
    let newQValue =
      (1 - this.alpha) * oldQValue +
      this.alpha * (actionReward + this.gamma * maxQValue);
    this.setQValue(state.name, action.name, newQValue);
    this.currentState = this.states[action.nextState];
    return this.currentState;
  }

  learn(steps) {
    steps = Math.max(1, steps || 0);
    while (steps--) {
      this.currentState = this.randomState();
      this.step();
    }
  }
}
