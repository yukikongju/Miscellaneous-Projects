import "./App.css";
import { useState, useEffect } from "react";
import Die from "./components/Die";
import { nanoid } from "nanoid";

function App() {
  const [dice, setDice] = useState(getNewDices());

  function getNewDices() {
    const newDice = Array.from({ length: 10 }, () => ({
      value: rollDie(),
      isHeld: false,
      id: nanoid(),
    }));
    return newDice;
  }

  function rollDie() {
    // returns random number from 0 to 9
    return Math.floor(Math.random() * 10);
  }

  function rollDice() {
    setDice((prevDice) => {
      const newDice = [];
      for (let i = 0; i < dice.length; i++) {
        const newDie = prevDice[i].isHeld
          ? prevDice[i]
          : { ...prevDice[i], value: rollDie() };
        newDice.push(newDie);
      }
      return newDice;
    });
  }

  function holdDice(id) {
    setDice((oldDice) =>
      oldDice.map((die) => {
        return die.id === id ? { ...die, isHeld: !die.isHeld } : die;
      })
    );
    console.log(dice);
  }

  useEffect(
    function () {
      // check is all dice are the same value
      const val = dice[0].value;
      for (let i = 0; i < dice.length; i++) {
        if (dice[i].value !== val || !dice[i].isHeld) {
          return;
        }
      }
      // if all values is the same
      alert("You won! All values are the same");
    },
    [dice]
  );

  const diceElements = dice.map((die) => (
    <Die
      key={die.id}
      value={die.value}
      isHeld={die.isHeld}
      id={die.id}
      holdDice={() => holdDice(die.id)}
    />
  ));

  return (
    <main>
      <div className="die-container">{diceElements}</div>
      <div className="bottom-div">
        <button className="button-roll" onClick={rollDice}>
          Roll the Dice
        </button>
      </div>
    </main>
  );
}

export default App;
