import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import * as employeService from '../../Service/EmployeService';

const initialState = {
  list: [],
  selectedEmploye: null,
  status: 'idle',
  error: null,
};

export const fetchEmployes = createAsyncThunk('employes/fetchEmployes', async () => {
  return await employeService.getEmployes();
});

export const addEmploye = createAsyncThunk('employes/addEmploye', async (employe) => {
  return await employeService.addEmploye(employe);
});

export const deleteEmploye = createAsyncThunk('employes/deleteEmploye', async (id) => {
  return await employeService.deleteEmploye(id);
});

export const updateEmploye = createAsyncThunk('employes/updateEmploye', async ({ id, employe }) => {
  return await employeService.updateEmploye(id, employe);
});


const employesSlice = createSlice({
  name: 'employes',
  initialState,
  reducers: {
    setSelectedEmploye: (state, action) => {
      state.selectedEmploye = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchEmployes.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchEmployes.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list = action.payload;
      })
      .addCase(fetchEmployes.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      })
      .addCase(addEmploye.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list.push(action.payload);
      })
      .addCase(deleteEmploye.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list = state.list.filter((employe) => employe.employeId !== action.payload);
      })
      .addCase(updateEmploye.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list = action.payload;
      })
  },
});

export default employesSlice.reducer;
export const { setSelectedEmploye } = employesSlice.actions;