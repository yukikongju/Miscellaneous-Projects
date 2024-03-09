import "./App.css";
import { useState, useEffect } from "react";
import Die from "./components/Die";
import { nanoid } from "nanoid";
import Confetti from "react-confetti";

function App() {
  const [dice, setDice] = useState(getNewDices());
  const [hasWin, setHasWin] = useState(false);

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

  function resetGame() {
    setDice(getNewDices());
    setHasWin(false);
  }

  useEffect(
    function () {
      // check is all dice are the same value and are held
      const allHeld = dice.every((die) => die.isHeld);
      const val = dice[0].value;
      const allSameValue = dice.every((die) => die.value === val);
      if (allHeld && allSameValue) {
        // alert("You won! All values are the same");
        console.log("You won! All values are the same");
        setHasWin(true);
      }
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
      {hasWin && <Confetti />}
      <div className="die-container">{diceElements}</div>
      <div className="bottom-div">
        {!hasWin && (
          <button className="button-roll" onClick={rollDice}>
            Roll the Dice
          </button>
        )}
        {hasWin && (
          <button className="button-newgame" onClick={resetGame}>
            Reset Game
          </button>
        )}
      </div>
    </main>
  );
}

export default App;
