import { configureStore, createSlice } from "@reduxjs/toolkit";

const languageSlice = createSlice({
  name: "language",
  initialState: "en",
  reducers: {
    setLanguage: (state, action) => action.payload,
  },
});

const { setLanguage } = languageSlice.actions;

const isLookingForBixiSlice = createSlice({
  name: "isLookingForBixi",
  initialState: true,
  reducers: {
    toggleIsLookingForBixi: (state) => !state,
  },
});

const { toggleIsLookingForBixi } = isLookingForBixiSlice.actions;

const store = configureStore({
  reducer: {
    language: languageSlice.reducer,
    isLookingForBixi: isLookingForBixiSlice.reducer,
  },
});

export default store;

// --- Language helpers

export function getLanguageStoreVariable() {
  return store.getState().language;
}

export function updateLanguageStoreVariable(newLanguage) {
  store.dispatch(setLanguage(newLanguage));
}

// --- isLookingForBixi helpers

export function getIsLookingForBixiStoreVariable() {
  return store.getState().isLookingForBixi;
}

export function toggleIsLookingForBixiStoreVariable() {
  store.dispatch(toggleIsLookingForBixi());
}
