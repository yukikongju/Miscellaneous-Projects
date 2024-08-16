import { configureStore, createSlice } from "@reduxjs/toolkit";

const languageSlice = createSlice({
  name: "language",
  initialState: "en",
  reducers: {
    setLanguage: (state, action) => action.payload,
  },
});

const isLookingForBixiSlice = createSlice({
  name: "isLookingForBixi",
  initialState: true,
  reducers: {
    toggleIsLookingForBixi: (state) => !state,
  },
});

const store = configureStore({
  reducer: {
    language: languageSlice.reducer,
    isLookingForBixi: isLookingForBixiSlice.reducer,
  },
});

export default store;

// --- Language helpers

export function getCurrentLanguage() {
  return store.getState().language;
}

export function updateStoreLanguage(newLanguage) {
  const { setLanguage } = languageSlice.actions;
  store.dispatch(setLanguage(newLanguage));
}

// --- isLookingForBixi helpers

export function getIsLookingForBixi() {
  return store.getState().isLookingForBixi;
}

export function toggleIsLookingForBixi() {
  const { toggleIsLookingForBixi } = isLookingForBixiSlice.actions;
  store.dispatch(toggleIsLookingForBixi());
}
