import { configureStore, createSlice } from "@reduxjs/toolkit";

const languageSlice = createSlice({
  name: "language",
  initialState: "en",
  reducers: {
    setLanguage: (state, action) => action.payload,
  },
});

export const { setLanguage } = languageSlice.actions;

const store = configureStore({
  reducer: {
    language: languageSlice.reducer,
  },
});

export default store;
